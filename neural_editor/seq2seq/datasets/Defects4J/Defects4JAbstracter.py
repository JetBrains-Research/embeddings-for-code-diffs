import os
import sys
from pathlib import Path
import subprocess
from subprocess import PIPE
from tqdm.auto import tqdm

# TODO: check that <ERROR> is not written


class Defects4JAbstracter:
    PREV_METHOD_FILENAME = 'prev_method.java'
    UPDATED_METHOD_FILENAME = 'updated_method.java'
    PREV_ABSTRACTED_METHOD_FILENAME = 'prev.txt'
    UPDATED_ABSTRACTED_METHOD_FILENAME = 'updated.txt'
    MAPPING_FILENAME = 'mapping.txt'

    def __init__(self, src2abs_jar: Path, idioms_path: Path) -> None:
        super().__init__()
        self.src2abs_jar = src2abs_jar
        self.idioms_path = idioms_path

    def abstract(self, root: Path) -> None:
        prev_method = root.joinpath(Defects4JAbstracter.PREV_METHOD_FILENAME)
        updated_method = root.joinpath(Defects4JAbstracter.UPDATED_METHOD_FILENAME)

        output_dir = root.joinpath('abstracted')
        output_dir.mkdir(exist_ok=True)

        prev_output_file = output_dir.joinpath('prev.txt')
        prev_output_file.touch(exist_ok=True)

        updated_output_file = output_dir.joinpath('updated.txt')
        updated_output_file.touch(exist_ok=True)

        abstraction_command = ['java', '-jar', str(self.src2abs_jar.absolute()),
                               'pair', 'method',
                               str(prev_method.absolute()), str(updated_method.absolute()),
                               str(prev_output_file.absolute()), str(updated_output_file.absolute()),
                               str(self.idioms_path.absolute())]

        completed_process = subprocess.run(abstraction_command, stdout=PIPE, stderr=PIPE)
        if completed_process.returncode != 0 \
                or not completed_process.stdout.startswith(b'Source Code Abstracted successfully!'):
            print()
            print(f'Something went wrong during abstraction of'
                  f'{str(prev_method.absolute())} and {str(updated_method.absolute())} pair.')
            print(f'Abstraction program exited with non-zero return code = {completed_process.returncode}')
            print(f'stdout: {completed_process.stdout}')
            print(f'stderr: {completed_process.stderr}')
            return
        mapping_old_file_path = output_dir.joinpath(f'{Defects4JAbstracter.PREV_ABSTRACTED_METHOD_FILENAME}.map')
        mapping_new_file_path = output_dir.joinpath(Defects4JAbstracter.MAPPING_FILENAME)
        mapping_old_file_path.rename(mapping_new_file_path)
        if prev_output_file.read_text() == '<ERROR>' or updated_output_file.read_text() == '<ERROR>':
            print()
            print(f'<ERROR> is written in abstracted file')
            print(f'Output dir: {str(output_dir.absolute())}')
            print(f'Prev file: {str(prev_method.absolute())}')
            print(f'Updated file: {str(updated_method.absolute())}')


def walk_through_dataset_and_abstract_method_pairs(dataset_root: Path, abstracter: Defects4JAbstracter) -> None:
    for (dir_path, _, filenames) in tqdm(list(os.walk(str(dataset_root.absolute())))):
        if Defects4JAbstracter.PREV_METHOD_FILENAME in filenames and \
                Defects4JAbstracter.UPDATED_METHOD_FILENAME in filenames:
            abstracter.abstract(Path(dir_path))


def main() -> None:
    if len(sys.argv) not in [3, 4]:
        print('Usage: <path to src2abs jar file to run abstraction> <root to dataset folder> <path to idioms ('
              'optional, if not provided then idioms in datasets/idioms.csv will be used, see readme of src2abs tool '
              'on GitHub for explanation what idioms mean)>')
        return
    src2abs_jar = Path(sys.argv[1])
    dataset_root = Path(sys.argv[2])
    idioms_path = Path('../idioms.csv')
    if len(sys.argv) == 4:
        idioms_path = Path(sys.argv[3])
    abstracter = Defects4JAbstracter(src2abs_jar, idioms_path)
    walk_through_dataset_and_abstract_method_pairs(dataset_root, abstracter)


if __name__ == "__main__":
    main()

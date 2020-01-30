import random
from distutils.dir_util import copy_tree
import sys
from pathlib import Path
from typing import List, Iterator, Tuple

SEED = 42375


def write_part(file_paths: Iterator[Tuple[Path, Path]], n: int) -> None:
    for prev_file_path, updated_file_path in file_paths:
        prev_lines: List[str] = prev_file_path.read_text(encoding='utf-8').splitlines(keepends=True)
        updated_lines: List[str] = updated_file_path.read_text(encoding='utf-8').splitlines(keepends=True)
        lines: List[Tuple[str, str]] = list(zip(prev_lines, updated_lines))
        random.Random(SEED).shuffle(lines)
        lines = lines[:n]
        unzipped_lines = [list(t) for t in zip(*lines)]
        prev_file_path.write_text(''.join(unzipped_lines[0]), encoding='utf-8')
        updated_file_path.write_text(''.join(unzipped_lines[1]), encoding='utf-8')


def take_part_from(dataset_type: str, dataset_path: Path, dataset_size: int):
    train_prev_file_paths: List[Path] = sorted(dataset_path.glob(f'**/{dataset_type}/prev.txt'))
    train_updated_file_paths: List[Path] = sorted(dataset_path.glob(f'**/{dataset_type}/updated.txt'))
    write_part(zip(train_prev_file_paths, train_updated_file_paths), dataset_size)


def create_small_dataset_for_testing(dataset_path: Path, train_dataset_size: int = 1024) -> None:
    new_dataset_path: Path = dataset_path.parent.joinpath(dataset_path.name + "_test")
    copy_tree(str(dataset_path.absolute()), str(new_dataset_path.absolute()))

    dataset_size = train_dataset_size / 0.8
    take_part_from('train', new_dataset_path, train_dataset_size)
    val_dataset_size = dataset_size * 0.1
    take_part_from('val', new_dataset_path, int(val_dataset_size))
    test_dataset_size = dataset_size * 0.1
    take_part_from('test', new_dataset_path, int(test_dataset_size))


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Args: <path_to_dataset> <train_dataset_size (optional, default=1024)>")
    elif len(sys.argv) == 3:
        create_small_dataset_for_testing(Path(sys.argv[1]), int(sys.argv[2]))
    else:
        create_small_dataset_for_testing(sys.argv[1])

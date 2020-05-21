from pathlib import Path
from typing import List, Tuple, Dict, Optional
from distutils.util import strtobool

from pydriller import RepositoryMining
from tqdm.auto import tqdm

from datasets.PatchNet.GitDiffPrevUpdatedGenerator import GitDiffPrevUpdatedGenerator


class Commit:
    def __init__(self, repository: str, commit_hash: str, lazy_initialization=False) -> None:
        super().__init__()
        self.repository = repository
        self.commit_hash = commit_hash
        self.prev_updated_generator = GitDiffPrevUpdatedGenerator()
        self.code = None if lazy_initialization else self.get_code()

    def get_prev(self) -> List[str]:
        return self.get_code_field('prev')

    def get_updated(self) -> List[str]:
        return self.get_code_field('updated')

    def get_code_field(self, field: str) -> List[str]:
        if self.code is None:
            self.code = self.get_code()
        return self.code[field]

    def get_code(self) -> Dict[str, List[str]]:
        commits = list(RepositoryMining(self.repository, single=self.commit_hash).traverse_commits())
        assert(len(commits) == 1)
        commit = commits[0]
        return self.prev_updated_generator.generate_prev_and_updated(commit)


class DataSample:
    def __init__(self, commit: Commit, stable: bool, idx: int) -> None:
        super().__init__()
        self.commit = commit
        self.stable = stable
        self.idx = idx


class PatchNetDataset:
    LINUX_REPOSITORY = 'https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git'

    def __init__(self, root: Path, description_filepath: Path, linux_repository_filepath: Optional[Path]) -> None:
        super().__init__()
        self.root = root
        self.description_filepath = description_filepath
        self.repository_path = str(linux_repository_filepath.absolute())
        self.data_samples = PatchNetDataset.extract_data_samples(self.description_filepath, self.repository_path)

    @staticmethod
    def extract_data_samples(description_filepath: Path, repository_path: str) -> List[DataSample]:
        examples_text_data = PatchNetDataset.get_examples_text_data(description_filepath)
        data_samples = []
        for idx, example_text_data in tqdm(list(enumerate(examples_text_data))):
            commit_hash = PatchNetDataset.extract_commit_hash_field(example_text_data)
            stable = PatchNetDataset.extract_stable_field(example_text_data)
            data_sample = DataSample(Commit(repository_path, commit_hash), stable, idx)
            data_samples.append(data_sample)
        return data_samples

    @staticmethod
    def extract_commit_hash_field(example_text_data: Tuple[str, str]) -> str:
        return example_text_data[0].split(': ')[1]

    @staticmethod
    def extract_stable_field(example_text_data: Tuple[str, str]) -> bool:
        return bool(strtobool(example_text_data[1].split(': ')[1]))

    @staticmethod
    def get_examples_text_data(description_filepath: Path) -> List[Tuple[str, str]]:
        description_lines = description_filepath.read_text().splitlines(keepends=False)
        return list(zip(description_lines[::2], description_lines[1::2]))

    def get_stable_patches(self) -> List[DataSample]:
        return [data_sample for data_sample in self.data_samples if data_sample.stable]

    def get_unstable_patches(self) -> List[DataSample]:
        return [data_sample for data_sample in self.data_samples if not data_sample.stable]

    def print_statistics(self) -> None:
        print(f'Dataset size: {len(self.data_samples)}')
        stable_patches = self.get_stable_patches()
        unstable_patches = self.get_unstable_patches()
        print(f'  Stable samples: {len(stable_patches)} ({round(len(stable_patches) / len(self.data_samples), 4)})')
        print(f'Unstable samples: {len(unstable_patches)} ({round(len(unstable_patches) / len(self.data_samples), 4)})')

    def write_data(self) -> None:
        prev_file_lines = [' '.join(data_sample.commit.get_prev()) for data_sample in self.data_samples]
        updated_file_lines = [' '.join(data_sample.commit.get_updated()) for data_sample in self.data_samples]
        trg_file_lines = [str(int(data_sample.stable)) for data_sample in self.data_samples]
        self.root.joinpath('prev.txt').write_text('\n'.join(prev_file_lines))
        self.root.joinpath('updated.txt').write_text('\n'.join(updated_file_lines))
        self.root.joinpath('trg.txt').write_text('\n'.join(trg_file_lines))


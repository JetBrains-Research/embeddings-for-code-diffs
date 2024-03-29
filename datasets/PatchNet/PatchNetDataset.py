import pickle
import time
from collections import Counter
from datetime import timedelta, datetime, timezone
from distutils.util import strtobool
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pydriller
from pydriller import RepositoryMining, ModificationType

from datasets.PatchNet.LevenshteinFilesPrevUpdatedGenerator import LevenshteinFilesPrevUpdatedGenerator


class Commit:
    def __init__(self, repository: str, commit_hash: str, commit: pydriller.Commit = None) -> None:
        super().__init__()
        self.repository = repository
        self.commit_hash = commit_hash
        self.prev_updated_generator = LevenshteinFilesPrevUpdatedGenerator()
        self.code = self.get_code(commit)

    def get_counter(self) -> Counter:
        return self.code[1]

    def get_prev(self) -> List[str]:
        return self.get_code_field('prev')

    def get_updated(self) -> List[str]:
        return self.get_code_field('updated')

    def get_code_field(self, field: str) -> List[str]:
        if self.code is None:
            self.code = self.get_code(None)
        return self.code[0][field]

    def get_code(self, commit: Optional[pydriller.Commit]) -> Tuple[Dict[str, List[str]], Counter]:
        if commit is None:
            commits = list(RepositoryMining(self.repository, single=self.commit_hash).traverse_commits())
            commit = commits[0]
        return self.prev_updated_generator.generate_prev_and_updated(commit)


class DataSample:
    def __init__(self, commit: Commit, stable: Optional[bool], idx: int) -> None:
        super().__init__()
        self.commit = commit
        self.stable = stable
        self.idx = idx


class PatchNetDataset:
    LINUX_REPOSITORY = 'https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git'
    SINCE_DATE = datetime(year=2009, month=6, day=23, hour=12, minute=48, second=1, tzinfo=timezone.utc)
    TO_DATE = datetime(year=2009, month=6, day=23, hour=22, minute=48, second=1, tzinfo=timezone.utc)
    MAX_DIFF_LENGTH = 100

    def __init__(self, root: Path, description_filepath: Optional[Path], linux_repository_filepath: Optional[Path]) -> None:
        super().__init__()
        self.root = root
        self.description_filepath = description_filepath
        self.repository_path = str(linux_repository_filepath.absolute()) if linux_repository_filepath is not None \
            else linux_repository_filepath
        if self.repository_path is not None:
            if self.description_filepath is not None:
                self.data_samples, self.tokens_counter = \
                    PatchNetDataset.extract_data_samples(self.description_filepath, self.repository_path)
            else:
                self.data_samples, self.tokens_counter = \
                    PatchNetDataset.extract_data_samples_for_pre_train(self.repository_path)
        else:
            self.data_samples = []
            self.tokens_counter = Counter()

    @staticmethod
    def extract_data_samples(description_filepath: Path, repository_path: str) -> Tuple[List[DataSample], Counter]:
        examples_text_data = PatchNetDataset.get_examples_text_data(description_filepath)
        data_samples = []
        counter = Counter()
        start = time.time()
        for idx, example_text_data in enumerate(examples_text_data):
            commit_hash = PatchNetDataset.extract_commit_hash_field(example_text_data)
            stable = PatchNetDataset.extract_stable_field(example_text_data)
            commit = Commit(repository_path, commit_hash)
            data_sample = DataSample(commit, stable, idx)
            data_samples.append(data_sample)
            counter += commit.get_counter()
            if (idx + 1) % 50 == 0:
                end = time.time()
                duration = end - start
                print(f'Processed {idx + 1} / {len(examples_text_data)} samples')
                print(f'Time elapsed: {str(timedelta(seconds=duration))}')
                start = end
        return data_samples, counter

    @staticmethod
    def extract_data_samples_for_pre_train(repository_path: str) -> Tuple[List[DataSample], Counter]:
        data_samples = []
        start = time.time()
        commits = list(RepositoryMining(repository_path, since=PatchNetDataset.SINCE_DATE, to=PatchNetDataset.TO_DATE,
                                        only_no_merge=True).traverse_commits())
        duration = time.time() - start
        print(f'PyDriller found commits since {PatchNetDataset.SINCE_DATE} to {PatchNetDataset.TO_DATE} for '
              f'{str(timedelta(seconds=duration))}\n')
        counter = Counter()
        start = time.time()
        for idx, commit in enumerate(commits):
            if not PatchNetDataset.is_commit_correct(commit):
                print(f'Commit with hash {commit.hash} is incorrect')
                continue
            commit = Commit(repository_path, commit.hash, commit)
            data_sample = DataSample(commit, False, idx)
            data_samples.append(data_sample)
            counter += commit.get_counter()
            if (idx + 1) % 50 == 0:
                end = time.time()
                duration = end - start
                print(f'Processed {idx + 1} / {len(commits)} samples')
                print(f'Time elapsed: {str(timedelta(seconds=duration))}')
                start = end
        return data_samples, counter

    @staticmethod
    def is_commit_correct(commit: pydriller.Commit):
        total_num_of_lines = 0
        for modification in commit.modifications:
            if modification.change_type == ModificationType.MODIFY:
                total_num_of_lines += len(modification.diff.splitlines())
            if not modification.filename.endswith('.c') and not modification.filename.endswith('.h'):
                return False
        return total_num_of_lines <= PatchNetDataset.MAX_DIFF_LENGTH

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
        prev_file_lines = [' '.join([t[1] for t in data_sample.commit.get_prev()]) for data_sample in self.data_samples]
        updated_file_lines = [' '.join([t[1] for t in data_sample.commit.get_updated()]) for data_sample in
                              self.data_samples]
        trg_file_lines = [str(int(data_sample.stable)) for data_sample in self.data_samples]
        ids_file_lines = [str(data_sample.idx) for data_sample in self.data_samples]
        commit_hashes_file_lines = [data_sample.commit.commit_hash for data_sample in self.data_samples]
        self.root.joinpath('prev.txt').write_text('\n'.join(prev_file_lines))
        self.root.joinpath('updated.txt').write_text('\n'.join(updated_file_lines))
        self.root.joinpath('trg.txt').write_text('\n'.join(trg_file_lines))
        self.root.joinpath('ids.txt').write_text('\n'.join(ids_file_lines))
        self.root.joinpath('commit_hashes.txt').write_text('\n'.join(commit_hashes_file_lines))
        with self.root.joinpath('tokens_counter.pkl').open('wb') as counter_file:
            pickle.dump(self.tokens_counter, counter_file)
        with self.root.joinpath('data_samples.pkl').open('wb') as data_sample_file:
            pickle.dump(self.data_samples, data_sample_file)

    def load(self) -> None:
        with self.root.joinpath('tokens_counter.pkl').open('rb') as counter_file:
            self.tokens_counter = pickle.load(counter_file)
        with self.root.joinpath('data_samples.pkl').open('rb') as counter_file:
            self.data_samples = pickle.load(counter_file)

    def filter_empty(self):
        size_before = len(self.data_samples)
        self.data_samples = [sample for sample in self.data_samples
                             if len(sample.commit.get_prev()) != 0 and len(sample.commit.get_updated()) != 0]
        size_after = len(self.data_samples)
        print(f'After removing of empties left: {size_after} / {size_before} = {size_after / size_before}')

    def remove_commits(self, commits_to_remove):
        size_before = len(self.data_samples)
        left_data_samples = []
        for sample in self.data_samples:
            if sample.commit.commit_hash not in commits_to_remove:
                left_data_samples.append(sample)
            else:
                for token in sample.commit.get_prev() + sample.commit.get_updated():
                    self.tokens_counter[token] -= 1
        self.data_samples = left_data_samples
        size_after = len(self.data_samples)
        print(f'Number of commits to remove: {len(commits_to_remove)}')
        print(f'After removing of commits: {size_after} / {size_before} = {size_after / size_before}')
        print(f'Number of removed commits from total: {size_before - size_after} / {len(commits_to_remove)} = '
              f'{(size_before - size_after) / len(commits_to_remove)}')

    def add(self, other_dataset):
        self.data_samples += other_dataset.data_samples
        self.tokens_counter += other_dataset.tokens_counter

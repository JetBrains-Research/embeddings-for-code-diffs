import sys
from pathlib import Path
from typing import Tuple, Dict

from more_itertools import unzip

from datasets.Jiang.GitDiffOutputProcessor import GitDiffOutputProcessor


def partition_dataset(data_root):
    pass


def generate_and_save_prev_and_updated_versions(data_root: Path) -> None:
    for data_dir in data_root.iterdir():
        diff_file = data_dir.joinpath('diff.txt')
        diffs = diff_file.read_text().splitlines()
        prev_and_updated = GitDiffOutputProcessor.get_prev_and_updated_for_diffs(diffs)
        prev, updated = unzip(prev_and_updated)
        data_dir.joinpath('prev.txt').write_text('\n'.join(prev))
        data_dir.joinpath('updated.txt').write_text('\n'.join(updated))


def strip_file(file: Path) -> None:
    stripped_diff = '\n'.join([line.strip() for line in file.read_text().splitlines()])
    file.write_text(stripped_diff)


def strip_dataset(data_root: Path) -> None:
    for data_dir in data_root.iterdir():
        strip_file(data_dir.joinpath('diff.txt'))
        strip_file(data_dir.joinpath('msg.txt'))


def main() -> None:
    if len(sys.argv) != 2:
        print("Correct Arguments: <path to folder containing Jiang dataset diffs and messages>")
        return
    data_root = Path(sys.argv[1])
    strip_dataset(data_root)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
from typing import Tuple, Dict, List
from distutils.dir_util import copy_tree

from more_itertools import unzip

from datasets.Jiang.DatasetFilter import DatasetFilter
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


def filter_dataset_inplace(data_root: Path) -> None:
    total_before = 0
    total_after = 0
    for data_dir in data_root.iterdir():
        diff_file = data_dir.joinpath('diff.txt')
        msg_file = data_dir.joinpath('msg.txt')
        diffs = diff_file.read_text().splitlines()
        msgs = msg_file.read_text().splitlines()
        datapoints = list(zip(diffs, msgs))
        filtered_datapoints = list(filter(lambda d: DatasetFilter.validate(d[0], d[1]), datapoints))
        filtered_diffs, filtered_msgs = unzip(filtered_datapoints)

        diff_file.write_text('\n'.join(filtered_diffs))
        msg_file.write_text('\n'.join(filtered_msgs))
        print(f'before: {len(datapoints)}, after: {len(filtered_datapoints)},'
              f' after / before: {len(filtered_datapoints) / len(datapoints)}')
        total_before += len(datapoints)
        total_after += len(filtered_datapoints)
    print(f'before: {total_before}, after: {total_after},'
          f' after / before: {total_after / total_before}')


def filter_dataset(data_root: Path) -> None:
    new_data_root = data_root.parent.joinpath('filtered_dataset')
    new_data_root.mkdir()
    copy_tree(str(data_root.absolute()), str(new_data_root.absolute()))
    filter_dataset_inplace(new_data_root)


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
    generate_and_save_prev_and_updated_versions(data_root)


if __name__ == "__main__":
    main()

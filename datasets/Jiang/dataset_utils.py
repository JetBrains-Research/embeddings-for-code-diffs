import os
import random
import sys
from distutils.dir_util import copy_tree
from pathlib import Path
import string

from more_itertools import unzip

from datasets.Jiang.DatasetFilter import DatasetFilter
from datasets.Jiang.GitDiffOutputProcessor import GitDiffOutputProcessor


def partition_dataset(data_root: Path) -> None:
    new_data_root = data_root.joinpath('partitioned')
    new_data_root.mkdir()
    neural_editor_root = new_data_root.joinpath('neural_editor')
    neural_editor_root.mkdir()
    commit_message_generator_root = new_data_root.joinpath('commit_message_generator')
    commit_message_generator_root.mkdir()
    for data_dir in data_root.iterdir():
        if data_dir.is_file() or data_dir.name == 'partitioned':
            continue
        neural_editor_dir = neural_editor_root.joinpath(data_dir.name)
        neural_editor_dir.mkdir()
        commit_message_generator_dir = commit_message_generator_root.joinpath(data_dir.name)
        commit_message_generator_dir.mkdir()

        diffs_file = data_dir.joinpath('diff.txt')
        msg_file = data_dir.joinpath('msg.txt')
        diffs_neural_editor = neural_editor_dir.joinpath('diff.txt')
        msg_neural_editor = neural_editor_dir.joinpath('msg.txt')
        diffs_commit_message_generator = commit_message_generator_dir.joinpath('diff.txt')
        msg_commit_message_generator = commit_message_generator_dir.joinpath('msg.txt')

        diffs = diffs_file.read_text().splitlines()
        msgs = msg_file.read_text().splitlines()
        assert len(diffs) == len(msgs)
        dataset_size = len(diffs)
        permutation = [i for i in range(dataset_size)]
        random.shuffle(permutation)

        division_id = int(len(permutation) / 2)
        neural_editor_permutation = permutation[:division_id]
        commit_message_generator_permutation = permutation[division_id:]

        diffs_neural_editor.write_text('\n'.join([diffs[ind] for ind in neural_editor_permutation]))
        msg_neural_editor.write_text('\n'.join([msgs[ind] for ind in neural_editor_permutation]))

        diffs_commit_message_generator.write_text('\n'.join([diffs[ind] for ind in commit_message_generator_permutation]))
        msg_commit_message_generator.write_text('\n'.join([msgs[ind] for ind in commit_message_generator_permutation]))


def generate_and_save_prev_and_updated_versions(data_root: Path) -> None:
    for data_dir in data_root.iterdir():
        if data_dir.is_file() or data_dir.name == 'partitioned':
            continue
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


def remove_punctuation(data_root: Path) -> None:
    remove_tokens(data_root, lambda t: t not in string.punctuation, 'removed_punctuation')


def remove_numbers(data_root: Path) -> None:
    def is_not_int(s):
        try:
            int(s)
            return False
        except ValueError:
            return True
    remove_tokens(data_root, is_not_int, 'removed_numbers')


def remove_tokens(data_root: Path, predicate, suffix: str) -> None:
    prev_name = data_root.name
    new_data_root = data_root.parent.joinpath(f'{prev_name}_{suffix}')
    new_data_root.mkdir()
    copy_tree(str(data_root.absolute()), str(new_data_root.absolute()))
    remove_tokens_inplace(new_data_root, predicate)


def remove_tokens_inplace(data_root: Path, predicate) -> None:
    for subdir, dirs, files in os.walk(str(data_root.absolute())):
        for msg_file in files:
            if msg_file != 'msg.txt':
                continue
            msg_file = Path(os.path.join(subdir, msg_file))
            msgs = msg_file.read_text().splitlines()
            msgs_without_punctuations = [' '.join([t for t in msg.split() if predicate(t)]) for msg in msgs]

            msg_file.write_text('\n'.join(msgs_without_punctuations))
            print(f'total number of tokens before: {sum([len(msg.split())for msg in msgs])}, '
                  f'total number of tokens after: {sum([len(msg.split())for msg in msgs_without_punctuations])},'
                  f' after / before: '
                  f'{sum([len(msg.split())for msg in msgs_without_punctuations]) / sum([len(msg.split()) for msg in msgs])}')


def main() -> None:
    if len(sys.argv) != 2:
        print("Correct Arguments: <path to folder containing Jiang dataset diffs and messages>")
        return
    data_root = Path(sys.argv[1])
    remove_numbers(data_root)


if __name__ == "__main__":
    main()

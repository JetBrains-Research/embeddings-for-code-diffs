import pickle
import random
import sys
from collections import Counter
from pathlib import Path

from datasets.PatchNet.PatchNetDataset import PatchNetDataset
from datasets.PatchNet.tokenizers import PygmentsCTokenizer
from datasets.dataset_utils import get_indices_for_train_val_test


def split_on_train_test_val():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']
    folder_names = ['train', 'val', 'test']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    train_indices, val_indices, test_indices = get_indices_for_train_val_test(len(data), ratios=(0.1, 0.1))
    indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}
    print(f'Train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')
    for folder_name in folder_names:
        path_to_write = root.joinpath(folder_name)
        path_to_write.mkdir(exist_ok=True)
        folder_indices = indices[folder_name]
        filenames_lines = {filename: [] for filename in filenames}
        for idx in folder_indices:
            data_sample = data[idx]
            for i, filename in enumerate(filenames_lines):
                filenames_lines[filename].append(data_sample[i])
        for filename, lines in filenames_lines.items():
            path_to_write.joinpath(filename).write_text('\n'.join(lines))


def cut_dataset(n, shuffle=False):
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    if shuffle:
        random.shuffle(data)
    data = data[:n]
    filenames_lines = {filename: [] for filename in filenames}
    for data_sample in data:
        for i, filename in enumerate(filenames_lines):
            filenames_lines[filename].append(data_sample[i])
    for filename, lines in filenames_lines.items():
        root.joinpath(filename).write_text('\n'.join(lines))


def partition_data():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']
    folder_names = ['neural_editor', 'predictor']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    ne_indices, predictor_indices, _ = get_indices_for_train_val_test(len(data), ratios=(0.5, 0))
    indices = {'neural_editor': ne_indices, 'predictor': predictor_indices}
    print(f'Neural editor: {len(ne_indices)}, predictor: {len(predictor_indices)}')
    for folder_name in folder_names:
        path_to_write = root.joinpath(folder_name)
        path_to_write.mkdir(exist_ok=True)
        folder_indices = indices[folder_name]
        filenames_lines = {filename: [] for filename in filenames}
        for idx in folder_indices:
            data_sample = data[idx]
            for i, filename in enumerate(filenames_lines):
                filenames_lines[filename].append(data_sample[i])
        for filename, lines in filenames_lines.items():
            path_to_write.joinpath(filename).write_text('\n'.join(lines))


def mine_dataset() -> None:
    if len(sys.argv) != 4:
        print('Usage: <root where to save processed data> <path to file with description of dataset> '
              '<path to local copy of linux git repository>')
        exit(1)
    root = Path(sys.argv[1])
    dataset_description_file = Path(sys.argv[2])
    linux_repository_filepath = Path(sys.argv[3])
    if not root.is_dir():
        print(f'No such directory: {root.absolute()}')
    if not dataset_description_file.is_file():
        print(f'No such file: {dataset_description_file.absolute()}')
        exit(1)
    if not linux_repository_filepath.is_dir():
        print(f'No such directory: {linux_repository_filepath.absolute()}')
    patch_net_dataset = PatchNetDataset(root, dataset_description_file, linux_repository_filepath)
    patch_net_dataset.print_statistics()
    patch_net_dataset.write_data()


def apply_tokenizer_again():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt']
    counter = Counter()
    tokenizer = PygmentsCTokenizer()
    for filename in filenames:
        lines = root.joinpath(filename).read_text().splitlines(keepends=False)
        lines_to_save = []
        for line in lines:
            tokens, line_counter = tokenizer.tokenize(line)
            lines_to_save.append(' '.join(tokens))
            counter += line_counter
        root.joinpath('filtered_' + filename).write_text('\n'.join(lines_to_save))
    with root.joinpath('identifier_names_counter.pkl').open('wb') as counter_file:
        pickle.dump(counter, counter_file)


if __name__ == "__main__":
    # cut_dataset(200, shuffle=False)
    # partition_data()
    # split_on_train_test_val()
    # mine_dataset()
    apply_tokenizer_again()

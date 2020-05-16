import json
import os
import sys
import random
from typing import Tuple

import numpy as np
from torchtext.data import Dataset, Field

from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.datasets.ClassifierDataset import ClassifierDataset


def split_train_val_test(root: str, train: float = 0.6, val: float = 0.2, test: float = 0.2) -> None:
    data_filenames = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    with open(os.path.join(root, 'train', 'train.jsonl'), 'w', encoding='utf-8-sig') as train_file, \
            open(os.path.join(root, 'val', 'val.jsonl'), 'w', encoding='utf-8-sig') as val_file, \
            open(os.path.join(root, 'test', 'test.jsonl'), 'w', encoding='utf-8-sig') as test_file:
        files = [train_file, val_file, test_file]
        for data_filename in data_filenames:
            with open(os.path.join(root, data_filename), mode='r', encoding='utf-8-sig') as data_file:
                for line in data_file:
                    file_to_write = files[np.random.choice(3, 1, p=[train, val, test])[0]]
                    file_to_write.write(line)


def concat_and_write_tokens_as_string_via_separator(root: str, sep: str = ' ') -> None:
    cases = ['train', 'val', 'test']
    for case in cases:
        with open(os.path.join(root, case, f'{case}.jsonl'), 'r', encoding='utf-8-sig') as case_json_file, \
                open(os.path.join(root, case, f'{case}_tokens_prev_text'), 'w', encoding='utf-8') as prev_tokens_text_file, \
                open(os.path.join(root, case, f'{case}_tokens_updated_text'), 'w', encoding='utf-8') as updated_tokens_text_file:
            for line in case_json_file:
                diff = json.loads(line)
                prev_tokens_string, updated_tokens_string = sep.join(diff['PrevCodeChunkTokens']), sep.join(diff['UpdatedCodeChunkTokens'])
                prev_tokens_text_file.write(prev_tokens_string + "\n")
                updated_tokens_text_file.write(updated_tokens_string + "\n")


def load_datasets(dataset_cls: Dataset.__class__, path: str, field: Field, config: Config,
                  train_reverse_examples_ratio: float,
                  train: str = 'train', val: str = 'val', test: str = 'test',
                  **kwargs) -> Tuple[Dataset, ...]:
    train_data: Dataset = dataset_cls(os.path.join(path, train), field, train_reverse_examples_ratio, config, **kwargs)
    val_data: Dataset = dataset_cls(os.path.join(path, val), field, 0, config, **kwargs)
    test_data: Dataset = dataset_cls(os.path.join(path, test), field, 0, config, **kwargs)
    return tuple(d for d in (train_data, val_data, test_data))


def prepare_dataset(dataset_root):
    split_train_val_test(dataset_root)
    concat_and_write_tokens_as_string_via_separator(dataset_root)


def take_part_from_dataset(dataset: Dataset, n: int) -> Dataset:
    import torchtext
    return torchtext.data.Dataset(dataset[:n], dataset.fields)


def take_subset_from_dataset(dataset: Dataset, indices) -> Dataset:
    import torchtext
    return torchtext.data.Dataset([dataset.examples[i] for i in indices], dataset.fields)


def get_indices_for_train_val_test(dataset_len: int, ratios=(0.1, 0.1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_size = int(dataset_len * ratios[1])
    val_size = int(dataset_len * ratios[0])
    indices = [i for i in range(dataset_len)]
    random.shuffle(indices)
    test_indices = indices[:test_size]
    val_indices = indices[test_size:(test_size + val_size)]
    train_indices = indices[(test_size + val_size):]
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def create_classifier_dataset(dataset, diffs_field, matrix, diff_example_ids):
    # TODO: fix that val, test and train intersects
    train_indices, val_indices, test_indices = get_indices_for_train_val_test(len(dataset))
    train_dataset = ClassifierDataset(dataset, train_indices, diff_example_ids, diffs_field, matrix)
    val_dataset = ClassifierDataset(dataset, val_indices, diff_example_ids, diffs_field, matrix)
    test_dataset = ClassifierDataset(dataset, test_indices, diff_example_ids, diffs_field, matrix)
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Args: <path_to_dataset>")
    prepare_dataset(sys.argv[1])

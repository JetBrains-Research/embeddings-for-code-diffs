import json
import os
import sys
from typing import Tuple, List

import numpy as np
from torchtext.data import Dataset, Field

from neural_editor.seq2seq.config import Config


def create_filter_predicate_on_length(max_length):
    def filter_predicate(example_data):
        for i, element in enumerate(example_data):
            if len(element) <= max_length:
                return True, None
            else:
                return False, \
                       f"{i}th element of example has length {len(element)} > {max_length}"
    return filter_predicate


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


def prepare_dataset(dataset_root):
    split_train_val_test(dataset_root)
    concat_and_write_tokens_as_string_via_separator(dataset_root)


def take_part_from_dataset(dataset: Dataset, n: int) -> Dataset:
    import torchtext
    return torchtext.data.Dataset(dataset[:n], dataset.fields)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Args: <path_to_dataset>")
    prepare_dataset(sys.argv[1])


def load_tufano_dataset(path: str, diffs_field: Field, config: Config) -> Tuple[Dataset, Dataset, Dataset]:
    from datasets.CodeChangesDataset import CodeChangesTokensDataset
    train_dataset, val_dataset, test_dataset = CodeChangesTokensDataset.load_datasets(path, diffs_field, config)
    return train_dataset, val_dataset, test_dataset
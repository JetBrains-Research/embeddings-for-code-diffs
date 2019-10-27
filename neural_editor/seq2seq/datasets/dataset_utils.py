import json
import os

import numpy as np

from neural_editor.seq2seq.train_config import CONFIG


def split_train_val_test(root, train=0.6, val=0.2, test=0.2):
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


def concat_and_write_tokens_as_string_via_separator(root, sep=' '):
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


def prepare_dataset():
    split_train_val_test(CONFIG['DATASET_ROOT'])
    concat_and_write_tokens_as_string_via_separator(CONFIG['DATASET_ROOT'])


if __name__ == "__main__":
    prepare_dataset()

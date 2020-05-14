import os
import sys
from typing import Tuple

import numpy as np
from torchtext import data
from torchtext.data import Field, Dataset

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.Batch import rebatch_iterator
from neural_editor.seq2seq.analyze import load_all
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.datasets.MatrixDataset import MatrixDataset
from neural_editor.seq2seq.datasets.dataset_utils import take_part_from_dataset
from neural_editor.seq2seq.train_utils import get_greedy_correct_predicted_examples


def get_matrix(model: EncoderDecoder, train_dataset: Dataset, diffs_field: Field, config: Config) \
        -> Tuple[np.ndarray, np.ndarray]:
    dataset_len = len(train_dataset)
    max_len = config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    sos_index = diffs_field.vocab.stoi[config['SOS_TOKEN']]
    eos_index = diffs_field.vocab.stoi[config['EOS_TOKEN']]

    n_neighbors = config['MATRIX_N_NEIGHBORS']
    if n_neighbors is None:
        n_neighbors = dataset_len
        diff_example_ids = np.tile(np.arange(dataset_len), (dataset_len, 1))
    else:
        model.set_training_data(train_dataset, pad_index)
        diff_example_ids_no_identity = model.get_neighbors(n_neighbors - 1)
        diff_example_ids = np.concatenate((np.arange(dataset_len).reshape(-1, 1), diff_example_ids_no_identity), axis=1)
        model.unset_training_data()
    matrix_dataset = MatrixDataset(train_dataset, diff_example_ids, diffs_field)
    print('Constructed matrix dataset, start predicting', flush=True)
    batched_data_iterator = rebatch_iterator(
        data.Iterator(matrix_dataset, batch_size=config['BATCH_SIZE'], train=False,
                      sort=False,
                      shuffle=False,
                      sort_within_batch=True,
                      sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                      device=config['DEVICE']), pad_index, config)
    model.unset_training_data()
    model.unset_edit_representation()
    correct_predicted_ids = get_greedy_correct_predicted_examples(batched_data_iterator, model,
                                                                  max_len, sos_index, eos_index)
    matrix = np.zeros((dataset_len, n_neighbors))
    if len(correct_predicted_ids) != 0:
        matrix[(correct_predicted_ids / matrix.shape[1]).astype(int), correct_predicted_ids % matrix.shape[1]] = 1
    return matrix, diff_example_ids


def write_matrix(matrix: np.ndarray, diff_example_ids: np.ndarray, config: Config) -> None:
    np.save(os.path.join(config['OUTPUT_PATH'], f'matrix.npy'), matrix)
    np.save(os.path.join(config['OUTPUT_PATH'], f'diff_example_ids.npy'), diff_example_ids)


def print_matrix_statistics(matrix: np.ndarray, diff_example_ids: np.ndarray):
    def print_accuracy(array, label):
        correct = (array.sum(axis=1) != 0).sum()
        print(f'{label}: {correct} / {array.shape[0]} = '
              f'{correct / array.shape[0]}')

    def print_non_zero_info(array, label):
        print(f'{label}: {(array != 0).sum()} / {array.shape[0] * array.shape[1]} = '
              f'{(array != 0).sum() / (array.shape[0] * array.shape[1])}')

    print(f'Matrix information')
    print(f'Shape: {matrix.shape}')
    print_non_zero_info(matrix, 'Non-zeros')
    print_non_zero_info(matrix[:, 1:], 'No-first column non-zeros')
    print_accuracy(matrix, 'Accuracy')
    print_accuracy(matrix[:, 1:], 'No-first column accuracy')
    top_k_values = [1, 3, 5, 10, 50]
    for top_k in top_k_values:
        print_accuracy(matrix[:, 1:(top_k + 1)], f'No-first column top-{top_k} accuracy')
    sample_idx = 20
    print(f'Matrix[:{sample_idx}, :{sample_idx}]')
    print(matrix[:sample_idx, :sample_idx])
    print(f'Ids[:{sample_idx}, :{sample_idx}]')
    print(diff_example_ids[:sample_idx, :sample_idx])


def load_matrix(root: str) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(os.path.join(root, f'matrix.npy')), \
           np.load(os.path.join(root, f'diff_example_ids.npy'))


def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("arguments: <results_root_dir> <is_test (optional, default false)>.")
        return
    results_root_dir = sys.argv[1]
    is_test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    model, (train_dataset, _, _, diffs_field), config = load_all(results_root_dir, is_test)
    matrix, diff_example_ids = get_matrix(model, train_dataset, diffs_field, config)
    print_matrix_statistics(matrix, diff_example_ids)
    write_matrix(matrix, diff_example_ids, config)


if __name__ == "__main__":
    main()

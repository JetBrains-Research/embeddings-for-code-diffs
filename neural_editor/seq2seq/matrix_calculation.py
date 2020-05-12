import os
import sys
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


def get_matrix(model: EncoderDecoder, train_dataset: Dataset, diffs_field: Field, config: Config) -> np.ndarray:
    matrix_dataset = MatrixDataset(train_dataset, diffs_field)
    max_len = config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    sos_index = diffs_field.vocab.stoi[config['SOS_TOKEN']]
    eos_index = diffs_field.vocab.stoi[config['EOS_TOKEN']]
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
    dataset_len = len(train_dataset)
    matrix = np.zeros((dataset_len, dataset_len))
    if len(correct_predicted_ids) != 0:
        matrix[(correct_predicted_ids / dataset_len).astype(int), correct_predicted_ids % dataset_len] = 1
    sample_idx = 20
    print(f'Matrix[:{sample_idx}, :{sample_idx}]')
    print(matrix[:sample_idx, :sample_idx])
    return matrix


def write_matrix(matrix: np.ndarray, config: Config) -> None:
    np.save(os.path.join(config['OUTPUT_PATH'], f'matrix.npy'), matrix)


def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("arguments: <results_root_dir> <is_test (optional, default false)>.")
        return
    results_root_dir = sys.argv[1]
    is_test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    model, (train_dataset, _, _, diffs_field), config = load_all(results_root_dir, is_test)
    matrix = get_matrix(model, train_dataset, diffs_field, config)
    write_matrix(matrix, config)


if __name__ == "__main__":
    main()

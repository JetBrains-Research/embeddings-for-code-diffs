import sys
from pathlib import Path

import numpy as np

from neural_editor.seq2seq.config import make_reproducible

SEED = 2354
NUM_WORDS = 11
LENGTH = 10
TRAINING_SIZE = 1600


def write_single_file(path: Path, data: np.array) -> None:
    """
    :param path: to file where to write data
    :param data: train data of shape [NumOfExamples, SeqLen]
    :return: nothing
    """
    lines = []
    prev_path = path.joinpath('prev.txt')
    updated_path = path.joinpath('updated.txt')
    for i in range(data.shape[0]):
        str_numbers = map(lambda n: str(n), data[i])
        lines.append(' '.join(str_numbers))
    prev_path.write_text('\n'.join(lines))
    updated_path.write_text('\n'.join(lines))


def generate_copy_task_dataset(root: Path, num_words: int, length: int, train_size: int) -> None:
    train_data = np.random.randint(1, num_words, size=(train_size, length))
    write_single_file(root.joinpath('train'), train_data)

    dataset_size = train_size / 0.8

    val_size = int(dataset_size * 0.1)
    val_data = np.random.randint(1, num_words, size=(val_size, length))
    write_single_file(root.joinpath('val'), val_data)

    test_size = int(dataset_size * 0.1)
    test_data = np.random.randint(1, num_words, size=(test_size, length))
    write_single_file(root.joinpath('test'), test_data)


if __name__ == "__main__":
    make_reproducible(SEED, False)
    if len(sys.argv) != 2:
        print("Args: <path_to_dataset>")
    generate_copy_task_dataset(Path(sys.argv[1]), NUM_WORDS, LENGTH, TRAINING_SIZE)
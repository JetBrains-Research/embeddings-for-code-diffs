"""
This file runs both training and evaluation
"""
import sys
from pathlib import Path

from neural_editor.seq2seq.analyze import test_model
from neural_editor.seq2seq.config import load_config
from neural_editor.seq2seq.matrix_calculation import get_matrix, write_matrix
from neural_editor.seq2seq.train import run_train


def main():
    is_test = len(sys.argv) > 1 and sys.argv[1] == 'test'
    config_path = None if len(sys.argv) < 3 else Path(sys.argv[2])
    config = load_config(is_test, config_path)
    print('\n====STARTING TRAINING====\n', end='')
    model, data = run_train(config)
    train_dataset, _, _, diffs_field = data
    print('\n====STARTING MATRIX CALCULATION====\n', end='')
    matrix, diff_example_ids = get_matrix(model, train_dataset, diffs_field, config)
    write_matrix(matrix, diff_example_ids, config)
    print('\n====STARTING EVALUATION====\n', end='')
    test_model(model, data, config)


if __name__ == "__main__":
    main()

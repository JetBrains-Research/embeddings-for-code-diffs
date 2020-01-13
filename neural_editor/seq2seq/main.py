"""
This file runs both training and evaluation
"""
import sys
from pathlib import Path

from neural_editor.seq2seq.analyze import test_model
from neural_editor.seq2seq.config import load_config
from neural_editor.seq2seq.train import run_train


def main():
    is_test = len(sys.argv) > 1 and sys.argv[1] == 'test'
    config_path = None if len(sys.argv) < 3 else Path(sys.argv[2])
    config = load_config(is_test, config_path)
    print('\n====STARTING TRAINING====\n', end='')
    model = run_train(config)
    print('\n====STARTING EVALUATION====\n', end='')
    test_model(model, config)


if __name__ == "__main__":
    main()

import os
import pickle
import pprint
import sys

import torch

from neural_editor.seq2seq.train import load_data, test
from neural_editor.seq2seq.train_utils import plot_perplexity


def load_model_and_test(results_root) -> None:
    _, _, test_dataset, diffs_field = load_data(verbose=True)
    model = torch.load(os.path.join(results_root, 'model.pt'))
    model.eval()
    test(model, test_dataset, diffs_field, print_every=-1)


def print_results(results_root: str) -> None:
    with open(os.path.join(results_root, 'config.pkl'), 'rb') as config_file:
        # noinspection PyPep8Naming
        # reason: we substitute global config with those which was during experiment
        # global CONFIG  # TODO: fix global config (this code doesn't change global config)
        config = pickle.load(config_file)
    pprint.pprint(config)
    # load_model_and_test(results_root)
    with open(os.path.join(results_root, 'train_perplexities.pkl'), 'rb') as train_file:
        train_perplexities = pickle.load(train_file)
    with open(os.path.join(results_root, 'val_perplexities.pkl'), 'rb') as val_file:
        val_perplexities = pickle.load(val_file)
    plot_perplexity([train_perplexities, val_perplexities], ['train', 'validation'])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("arguments: <results_root_dir>. You passed more than 1 argument")
    results_root_dir = sys.argv[1]
    print_results(results_root_dir)

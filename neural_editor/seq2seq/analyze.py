import os
import pickle
import pprint
import sys

import torch

from neural_editor.seq2seq.test import one_shot_learning, visualization
from neural_editor.seq2seq.test_utils import plot_perplexity, load_defects4j_dataset, output_accuracy_on_defects4j
from neural_editor.seq2seq.train import load_data


def load_model_and_test(results_root: str) -> None:
    train_dataset, val_dataset, test_dataset, diffs_field = load_data(verbose=True)
    defects4j_dataset, defects4j_classes = load_defects4j_dataset(diffs_field)
    model = torch.load(os.path.join(results_root, 'model.pt'), map_location=torch.device('cpu'))
    model.eval()
    visualization(model, defects4j_dataset, defects4j_classes, diffs_field)
    one_shot_learning(model, defects4j_dataset, defects4j_classes, diffs_field)
    output_accuracy_on_defects4j(model, defects4j_dataset, diffs_field)


def print_results(results_root: str) -> None:
    with open(os.path.join(results_root, 'config.pkl'), 'rb') as config_file:
        # noinspection PyPep8Naming
        # reason: we substitute global config with those which was during experiment
        # global CONFIG  # TODO: fix global config (this code doesn't change global config)
        config = pickle.load(config_file)
    pprint.pprint(config)
    load_model_and_test(results_root)
    with open(os.path.join(results_root, 'train_perplexities.pkl'), 'rb') as train_file:
        train_perplexities = pickle.load(train_file)
    with open(os.path.join(results_root, 'val_perplexities.pkl'), 'rb') as val_file:
        val_perplexities = pickle.load(val_file)
    plot_perplexity([train_perplexities, val_perplexities], ['train', 'validation'])


def main() -> None:
    if len(sys.argv) != 2:
        print("arguments: <results_root_dir>. You passed more or less than 1 argument")
    results_root_dir = sys.argv[1]
    print_results(results_root_dir)


if __name__ == "__main__":
    main()

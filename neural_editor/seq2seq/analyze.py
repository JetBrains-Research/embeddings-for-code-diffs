import os
import pickle
import pprint
import sys
from pathlib import Path

import torch
from torchtext.data import Dataset

from neural_editor.seq2seq.config import Config, load_config
from neural_editor.seq2seq.experiments.OneShotLearning import OneShotLearning
from neural_editor.seq2seq.test import visualization_for_classified_dataset, visualization_for_unclassified_dataset
from neural_editor.seq2seq.test_utils import plot_perplexity, load_defects4j_dataset, output_accuracy_on_defects4j, \
    load_tufano_labeled_dataset
from neural_editor.seq2seq.train import load_data


def load_model_and_test(results_root: str, config: Config) -> None:
    train_dataset, val_dataset, test_dataset, diffs_field = load_data(verbose=True, config=config)
    tufano_labeled_dataset, tufano_labeled_classes = load_tufano_labeled_dataset(diffs_field, config)
    defects4j_dataset, defects4j_classes = load_defects4j_dataset(diffs_field, config)
    model = torch.load(os.path.join(results_root, 'model.pt'), map_location=torch.device('cpu'))

    one_shot_learning_experiment = OneShotLearning(model, diffs_field, config)

    model.eval()
    with torch.no_grad():
        one_shot_learning_experiment.conduct(tufano_labeled_dataset, tufano_labeled_classes,
                                             'Tufano Labeled Code Changes')
        visualization_for_classified_dataset(model, tufano_labeled_dataset, tufano_labeled_classes, diffs_field, config)

        visualization_for_classified_dataset(model, defects4j_dataset, defects4j_classes,
                                             diffs_field, config)
        visualization_for_unclassified_dataset(model, Dataset(train_dataset[:500], train_dataset.fields),
                                               diffs_field, config)
        visualization_for_unclassified_dataset(model, Dataset(val_dataset[:500], val_dataset.fields),
                                               diffs_field, config)
        visualization_for_unclassified_dataset(model, Dataset(test_dataset[:500], test_dataset.fields),
                                               diffs_field, config)
        one_shot_learning_experiment.conduct(defects4j_dataset, defects4j_classes, 'Defects4J')
        output_accuracy_on_defects4j(model, defects4j_dataset, diffs_field, config)


def print_results(results_root: str, config: Config) -> None:
    pprint.pprint(config.get_config())
    load_model_and_test(results_root, config)
    with open(os.path.join(results_root, 'train_perplexities.pkl'), 'rb') as train_file:
        train_perplexities = pickle.load(train_file)
    with open(os.path.join(results_root, 'val_perplexities.pkl'), 'rb') as val_file:
        val_perplexities = pickle.load(val_file)
    plot_perplexity([train_perplexities, val_perplexities], ['train', 'validation'])


def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("arguments: <results_root_dir> <is_test (optional, default false)>.")
    results_root_dir = sys.argv[1]
    is_test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    config_path = Path(results_root_dir).joinpath('config.pkl')
    config = load_config(is_test, config_path)
    print_results(results_root_dir, config)


if __name__ == "__main__":
    main()

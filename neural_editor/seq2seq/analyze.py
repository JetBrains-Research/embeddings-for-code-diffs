import os
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config, load_config
from neural_editor.seq2seq.datasets.dataset_utils import take_part_from_dataset
from neural_editor.seq2seq.experiments.AccuracyCalculation import AccuracyCalculation
from neural_editor.seq2seq.experiments.EditRepresentationVisualization import EditRepresentationVisualization
from neural_editor.seq2seq.experiments.OneShotLearning import OneShotLearning
from neural_editor.seq2seq.test_utils import load_defects4j_dataset, load_tufano_labeled_dataset
from neural_editor.seq2seq.train import load_data


def measure_experiment_time(func) -> None:
    start = time.time()
    func()
    end = time.time()
    print(f'Duration: {str(timedelta(seconds=end - start))}')
    print()


def test_model(model: EncoderDecoder, config: Config) -> None:
    train_dataset, val_dataset, test_dataset, diffs_field = load_data(verbose=True, config=config)
    tufano_labeled_dataset, tufano_labeled_classes = load_tufano_labeled_dataset(diffs_field, config)
    defects4j_dataset, defects4j_classes = load_defects4j_dataset(diffs_field, config)

    one_shot_learning_experiment = OneShotLearning(model, diffs_field, config)
    accuracy_calculation_experiment = AccuracyCalculation(model, diffs_field, config)
    visualization_experiment = EditRepresentationVisualization(model, diffs_field, config)

    model.eval()
    model.unset_edit_representation()
    with torch.no_grad():
        # Visualization of data
        measure_experiment_time(
            lambda: visualization_experiment.conduct(tufano_labeled_dataset,
                                                     'tufano_labeled_2d_representations.png',
                                                     classes=tufano_labeled_classes)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(defects4j_dataset,
                                                     'defects4j_2d_representations.png',
                                                     classes=defects4j_classes)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(take_part_from_dataset(test_dataset, 300),
                                                     'test300_2d_representations.png', classes=None)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(take_part_from_dataset(val_dataset, 300),
                                                     'val300_2d_representations.png', classes=None)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(take_part_from_dataset(train_dataset, 300),
                                                     'train300_2d_representations.png', classes=None)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(test_dataset,
                                                     'test_2d_representations.png', classes=None)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(val_dataset,
                                                     'val_2d_representations.png', classes=None)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(take_part_from_dataset(train_dataset, 5000),
                                                     'train5000_2d_representations.png', classes=None)
        )

        # Accuracy
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(tufano_labeled_dataset,
                                                            'Tufano Labeled Code Changes')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(defects4j_dataset, 'Defects4J')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(test_dataset, 300), 'Test dataset 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(val_dataset, 300), 'Validation dataset 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(train_dataset, 300), 'Train dataset 300')
        )

        # One shot learning
        measure_experiment_time(
            lambda: one_shot_learning_experiment.conduct(tufano_labeled_dataset, tufano_labeled_classes,
                                                         'Tufano Labeled Code Changes')
        )
        measure_experiment_time(
            lambda: one_shot_learning_experiment.conduct(defects4j_dataset, defects4j_classes, 'Defects4J')
        )

        # Long execution
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                test_dataset, 'Test dataset all')
        )


def print_results(results_root: str, config: Config) -> None:
    pprint.pprint(config.get_config())
    model = torch.load(os.path.join(results_root, 'model.pt'), map_location=config['DEVICE'])
    test_model(model, config)


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

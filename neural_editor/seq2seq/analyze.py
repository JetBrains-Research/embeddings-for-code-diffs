import os
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.classifier_utils import load_classifier
from neural_editor.seq2seq.config import Config, load_config
from neural_editor.seq2seq.datasets.dataset_utils import take_part_from_dataset
from neural_editor.seq2seq.experiments.AccuracyCalculation import AccuracyCalculation
from neural_editor.seq2seq.experiments.EditRepresentationVisualization import EditRepresentationVisualization
from neural_editor.seq2seq.experiments.NearestNeighbor import NearestNeighbor
from neural_editor.seq2seq.experiments.NearestNeighborAccuracyOnLabeledData import NearestNeighborAccuracyOnLabeledData
from neural_editor.seq2seq.experiments.OneShotLearning import OneShotLearning
from neural_editor.seq2seq.test_utils import load_defects4j_dataset, load_labeled_dataset
from neural_editor.seq2seq.train import load_data, load_tufano_dataset
from neural_editor.seq2seq.train_utils import make_model


def measure_experiment_time(func) -> Any:
    start = time.time()
    ret = func()
    end = time.time()
    print(f'Duration: {str(timedelta(seconds=end - start))}')
    print()
    return ret


def test_model(model: EncoderDecoder, data, config: Config) -> None:
    train_dataset, val_dataset, test_dataset, diffs_field = data
    pad_index = diffs_field.vocab.stoi[config['PAD_TOKEN']]

    tufano_labeled_0_50_dataset, tufano_labeled_0_50_classes = \
        load_labeled_dataset(config['TUFANO_LABELED_0_50_PATH'], diffs_field, config)
    tufano_labeled_50_100_dataset, tufano_labeled_50_100_classes = \
        load_labeled_dataset(config['TUFANO_LABELED_50_100_PATH'], diffs_field, config)
    defects4j_dataset, defects4j_classes = load_defects4j_dataset(diffs_field, config)
    tufano_bug_fixes_0_50_dataset_train, tufano_bug_fixes_0_50_dataset_val, tufano_bug_fixes_0_50_dataset_test = \
        load_tufano_dataset(config['TUFANO_BUG_FIXES_0_50_PATH'], diffs_field, config)
    tufano_bug_fixes_50_100_dataset_train, tufano_bug_fixes_50_100_dataset_val, tufano_bug_fixes_50_100_dataset_test = \
        load_tufano_dataset(config['TUFANO_BUG_FIXES_50_100_PATH'], diffs_field, config)
    tufano_code_changes_0_50_dataset_train, tufano_code_changes_0_50_dataset_val, tufano_code_changes_0_50_dataset_test = \
        load_tufano_dataset(config['TUFANO_CODE_CHANGES_0_50_PATH'], diffs_field, config)
    tufano_code_changes_50_100_dataset_train, tufano_code_changes_50_100_dataset_val, tufano_code_changes_50_100_dataset_test = \
        load_tufano_dataset(config['TUFANO_CODE_CHANGES_50_100_PATH'], diffs_field, config)

    one_shot_learning_experiment = OneShotLearning(model, diffs_field, config)
    accuracy_calculation_experiment = AccuracyCalculation(model, diffs_field, train_dataset, config)
    visualization_experiment = EditRepresentationVisualization(model, diffs_field, config)
    nearest_neighbor_experiment = NearestNeighbor(model, pad_index, config)
    nearest_neighbor_accuracy_on_labeled_data_experiment = \
        NearestNeighborAccuracyOnLabeledData(nearest_neighbor_experiment, config)

    model.eval()
    model.unset_edit_representation()
    model.unset_training_data()
    with torch.no_grad():
        # Visualization of data
        """
        measure_experiment_time(
            lambda: visualization_experiment.conduct(tufano_labeled_0_50_dataset,
                                                     'tufano_labeled_0_50_2d_representations.png',
                                                     classes=tufano_labeled_0_50_classes)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(tufano_labeled_0_50_dataset,
                                                     'tufano_labeled_0_50_2d_representations_8_threshold.png',
                                                     classes=tufano_labeled_0_50_classes,
                                                     threshold=8)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(tufano_labeled_50_100_dataset,
                                                     'tufano_labeled_50_100_2d_representations.png',
                                                     classes=tufano_labeled_50_100_classes)
        )
        measure_experiment_time(
            lambda: visualization_experiment.conduct(tufano_labeled_50_100_dataset,
                                                     'tufano_labeled_50_100_2d_representations_8_threshold.png',
                                                     classes=tufano_labeled_50_100_classes,
                                                     threshold=8)
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
        """
        # Nearest neighbor accuracy on labeled data
        measure_experiment_time(
            lambda: nearest_neighbor_accuracy_on_labeled_data_experiment.conduct(
                dataset=tufano_labeled_0_50_dataset,
                classes=tufano_labeled_0_50_classes,
                dataset_label='Tufano Labeled 0 50 Code Changes')
        )
        measure_experiment_time(
            lambda: nearest_neighbor_accuracy_on_labeled_data_experiment.conduct(
                dataset=tufano_labeled_50_100_dataset,
                classes=tufano_labeled_50_100_classes,
                dataset_label='Tufano Labeled 50 100 Code Changes')
        )

        # Nearest neighbor between train and test 300
        measure_experiment_time(
            lambda: nearest_neighbor_experiment.conduct(dataset_train=train_dataset,
                                                        dataset_test=take_part_from_dataset(test_dataset, 300),
                                                        dataset_label='Train dataset Test dataset 300')
        )

        # Accuracy
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(tufano_labeled_0_50_dataset,
                                                            'Tufano Labeled 0 50 Code Changes')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(tufano_labeled_50_100_dataset,
                                                            'Tufano Labeled 50 100 Code Changes')
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
            lambda: one_shot_learning_experiment.conduct(tufano_labeled_0_50_dataset, tufano_labeled_0_50_classes,
                                                         'Tufano Labeled 0 50 Code Changes')
        )
        measure_experiment_time(
            lambda: one_shot_learning_experiment.conduct(tufano_labeled_50_100_dataset, tufano_labeled_50_100_classes,
                                                         'Tufano Labeled 50 100 Code Changes')
        )
        measure_experiment_time(
            lambda: one_shot_learning_experiment.conduct(defects4j_dataset, defects4j_classes, 'Defects4J')
        )

        print('Starting long experiments', flush=True)

        # Whole test accuracy
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                test_dataset, 'Test dataset all')
        )

        # Nearest neighbor between train and test, train and train
        #measure_experiment_time(
        #    lambda: nearest_neighbor_experiment.conduct(dataset_train=train_dataset,
        #                                                dataset_test=test_dataset,
        #                                                dataset_label='Train dataset Test dataset')
        #)
        #measure_experiment_time(
        #    lambda: nearest_neighbor_experiment.conduct(dataset_train=train_dataset,
        #                                                dataset_test=None,
        #                                                dataset_label='Train dataset None')
        #)

        print('Starting all Tufano variations experiments', flush=True)

        # Tufano accuracy evaluation
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_bug_fixes_0_50_dataset_train, 300),
                'Tufano bug fixes 0 50 dataset train 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_bug_fixes_0_50_dataset_val, 300),
                'Tufano bug fixes 0 50 dataset val 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_bug_fixes_0_50_dataset_test, 300),
                'Tufano bug fixes 0 50 dataset test 300')
        )

        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_bug_fixes_50_100_dataset_train, 300),
                'Tufano bug fixes 50 100 dataset train 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_bug_fixes_50_100_dataset_val, 300),
                'Tufano bug fixes 50 100 dataset val 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_bug_fixes_50_100_dataset_test, 300),
                'Tufano bug fixes 50 100 dataset test 300')
        )

        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_code_changes_0_50_dataset_train, 300),
                'Tufano code changes 0 50 dataset train 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_code_changes_0_50_dataset_val, 300),
                'Tufano code changes 0 50 dataset val 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_code_changes_0_50_dataset_test, 300),
                'Tufano code changes 0 50 dataset test 300')
        )

        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_code_changes_50_100_dataset_train, 300),
                'Tufano code changes 50 100 dataset train 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_code_changes_50_100_dataset_val, 300),
                'Tufano code changes 50 100 dataset val 300')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                take_part_from_dataset(tufano_code_changes_50_100_dataset_test, 300),
                'Tufano code changes 50 100 dataset test 300')
        )

        # All test data Tufano dataset evaluation
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                tufano_bug_fixes_0_50_dataset_test, 'Tufano bug fixes 0 50 dataset test')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                tufano_bug_fixes_50_100_dataset_test, 'Tufano bug fixes 50 100 dataset test')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                tufano_code_changes_0_50_dataset_test, 'Tufano code changes 0 50 dataset test')
        )
        measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                tufano_code_changes_50_100_dataset_test, 'Tufano code changes 50 100 dataset test')
        )


def load_model(results_root: str, vocab_size: int, config: Config) -> nn.Module:
    model = make_model(vocab_size,
                       edit_representation_size=config['EDIT_REPRESENTATION_SIZE'],
                       emb_size=config['WORD_EMBEDDING_SIZE'],
                       hidden_size_encoder=config['ENCODER_HIDDEN_SIZE'],
                       hidden_size_decoder=config['DECODER_HIDDEN_SIZE'],
                       num_layers=config['NUM_LAYERS'],
                       dropout=config['DROPOUT'],
                       use_bridge=config['USE_BRIDGE'],
                       config=config)
    model.load_state_dict(torch.load(os.path.join(results_root, 'model_state_dict_best_on_validation.pt'),
                                     map_location=config['DEVICE']))
    return model


def load_all(results_root_dir, is_test):
    def change_config():
        pass
        #config._CONFIG['EARLY_STOPPING_ROUNDS_CLASSIFIER'] = 100
        #config._CONFIG['METRIC'] = 'minkowski'
        #config._CONFIG['EVALUATION_ROUNDS_CLASSIFIER'] = 100
        #config._CONFIG['CLASSIFIER_EARLY_STOPPING_NO_EPOCHS'] = True
        #config._CONFIG['CLASSIFIER_FILENAME'] = 'model_state_dict_best_on_validation_classifier.pt'
        # config._CONFIG['OUTPUT_PATH'] = results_root_dir

    config_path = Path(results_root_dir).joinpath('config.pkl')
    config = load_config(is_test, config_path)
    change_config()
    pprint.pprint(config.get_config())
    data = load_data(verbose=True, config=config)
    model = load_model(results_root_dir, len(data[3].vocab), config)
    if config['CLASSIFIER_FILENAME'] is not None and \
            os.path.isfile(os.path.join(results_root_dir, config['CLASSIFIER_FILENAME'])):
        classifier = \
            load_classifier(os.path.join(results_root_dir, config['CLASSIFIER_FILENAME']), len(data[3].vocab), config)
        model.set_classifier(classifier)
    return model, data, config


def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("arguments: <results_root_dir> <is_test (optional, default false)>.")
    results_root_dir = sys.argv[1]
    is_test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    model, data, config = load_all(results_root_dir, is_test)
    test_model(model, data, config)


if __name__ == "__main__":
    main()

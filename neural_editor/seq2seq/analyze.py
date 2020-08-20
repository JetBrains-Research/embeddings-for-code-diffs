import os
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, List

import torch
from torchtext.data import Field

from datasets.CodeChangesDataset import CodeChangesTokensDataset
from datasets.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from datasets.dataset_utils import take_part_from_dataset
from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config, load_config
from neural_editor.seq2seq.experiments.AccuracyCalculation import AccuracyCalculation
from neural_editor.seq2seq.experiments.BleuCalculation import BleuCalculation
from neural_editor.seq2seq.experiments.EditRepresentationVisualization import EditRepresentationVisualization
from neural_editor.seq2seq.experiments.NearestNeighbors import NearestNeighbors, FeaturesType
from neural_editor.seq2seq.experiments.OneShotLearning import OneShotLearning
from neural_editor.seq2seq.test_utils import load_defects4j_dataset, load_labeled_dataset, save_predicted


def measure_experiment_time(func) -> Any:
    start = time.time()
    ret = func()
    end = time.time()
    print(f'Duration: {str(timedelta(seconds=end - start))}')
    print()
    return ret


def test_commit_message_generation_model(model: EncoderDecoder, config: Config, diffs_field: Field, greedy: bool) -> None:
    train_dataset, val_dataset, test_dataset, fields_commit = \
        CommitMessageGenerationDataset.load_data(diffs_field, config['VERBOSE'], config)
    train_dataset_test_size_part = take_part_from_dataset(train_dataset, len(test_dataset))
    accuracy_calculation_experiment = AccuracyCalculation(model, fields_commit[1],
                                                          config['MSG_MAX_LEN'] + 1, greedy, config)
    bleu_calculation_experiment = BleuCalculation(config)
    suffix_for_saving_predictions = 'commit_message_generator' + ('_greedy' if greedy else '')

    model.eval()
    model.unset_edit_representation()
    with torch.no_grad():
        test_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(test_dataset, 'Test dataset')
        )
        save_predicted(test_max_top_k_predicted, dataset_name=f'test_dataset_{suffix_for_saving_predictions}', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(test_max_top_k_predicted, test_dataset,
                                                        'Test dataset')
        )

        val_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(val_dataset, 'Validation dataset')
        )
        save_predicted(val_max_top_k_predicted, dataset_name=f'val_dataset_{suffix_for_saving_predictions}', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(val_max_top_k_predicted, val_dataset,
                                                        'Validation dataset')
        )

        train_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(train_dataset_test_size_part,
                                                            f'Train dataset (test size approximation)')
        )
        save_predicted(train_max_top_k_predicted, dataset_name=f'train_dataset_{suffix_for_saving_predictions}', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(train_max_top_k_predicted, train_dataset_test_size_part,
                                                        'Train dataset (test size approximation)')
        )


def test_neural_editor_model(model: EncoderDecoder, config: Config) -> Field:
    train_dataset, val_dataset, test_dataset, diffs_field = \
        CodeChangesTokensDataset.load_data(verbose=True, config=config)
    if not config['ANALYZE_NE']:
        print('Neural editor will not be tested because edit representations are not used.')
        return diffs_field
    train_dataset_test_size_part = take_part_from_dataset(train_dataset, len(test_dataset))
    tufano_labeled_0_50_dataset, tufano_labeled_0_50_classes = \
        load_labeled_dataset(config['TUFANO_LABELED_0_50_PATH'], diffs_field, config)
    tufano_labeled_50_100_dataset, tufano_labeled_50_100_classes = \
        load_labeled_dataset(config['TUFANO_LABELED_50_100_PATH'], diffs_field, config)
    defects4j_dataset, defects4j_classes = load_defects4j_dataset(diffs_field, config)

    accuracy_calculation_experiment = AccuracyCalculation(model, diffs_field, config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1,
                                                          greedy=False, config=config)
    bleu_calculation_experiment = BleuCalculation(config)
    one_shot_learning_experiment = OneShotLearning(model, diffs_field, config)

    model.eval()
    model.unset_edit_representation()
    with torch.no_grad():
        # Accuracy
        measure_experiment_time(
            lambda: one_shot_learning_experiment.conduct(tufano_labeled_0_50_dataset, tufano_labeled_0_50_classes,
                                                         'Tufano Labeled 0 50 Code Changes')
        )
        measure_experiment_time(
            lambda: one_shot_learning_experiment.conduct(tufano_labeled_50_100_dataset, tufano_labeled_50_100_classes,
                                                         'Tufano Labeled 50 100 Code Changes')
        )

        test_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(test_dataset, 'Test dataset')
        )
        save_predicted(test_max_top_k_predicted, dataset_name='test_dataset_neural_editor', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(test_max_top_k_predicted, test_dataset,
                                                        'Test dataset'))

        val_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(val_dataset, 'Validation dataset')
        )
        save_predicted(val_max_top_k_predicted, dataset_name='val_dataset_neural_editor', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(val_max_top_k_predicted, val_dataset,
                                                        'Validation dataset'))

        train_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(
                train_dataset_test_size_part, f'Train dataset (test size approximation)')
        )
        save_predicted(train_max_top_k_predicted, dataset_name='train_dataset_neural_editor', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(train_max_top_k_predicted, train_dataset_test_size_part,
                                                        'Train dataset (test size approximation)'))

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

    return diffs_field


def test_nearest_neighbors(model: EncoderDecoder, config: Config) -> None:
    train_dataset_ne, val_dataset_ne, test_dataset_ne, diffs_field = \
        CodeChangesTokensDataset.load_data(verbose=True, config=config)
    train_dataset_cmg, val_dataset_cmg, test_dataset_cmg, fields_commit = \
        CommitMessageGenerationDataset.load_data(diffs_field, config['VERBOSE'], config)
    nearest_neighbors_experiment = NearestNeighbors(model, fields_commit[1], config)

    model.eval()
    model.unset_edit_representation()
    with torch.no_grad():
        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([val_dataset_cmg],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'val_cmg')
        )

        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([val_dataset_ne],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'val_ne')
        )

        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([train_dataset_cmg],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'train_cmg')
        )

        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([train_dataset_ne],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'train_ne')
        )

        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([train_dataset_ne, val_dataset_ne],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'train_ne_and_val_ne')
        )

        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([train_dataset_ne, val_dataset_ne, train_dataset_cmg],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'train_ne_and_val_ne_and_train_cmg')
        )

        measure_experiment_time(
            lambda: nearest_neighbors_experiment.conduct([train_dataset_ne, val_dataset_ne, train_dataset_cmg, val_dataset_cmg],
                                                         {'val_dataset_cmg': val_dataset_cmg,
                                                          'test_dataset_cmg': test_dataset_cmg},
                                                         FeaturesType.SRC_AND_EDIT,
                                                         'train_ne_and_val_ne_and_train_cmg_and_val_cmg')
        )


def print_results(results_root: str, config: Config) -> None:
    pprint.pprint(config.get_config())
    neural_editor = None
    if config['USE_EDIT_REPRESENTATION']:
        neural_editor = torch.load(os.path.join(results_root, 'model_best_on_validation_neural_editor.pt'),
                                   map_location=config['DEVICE'])
    print('\n====STARTING NEAREST NEIGHBORS EVALUATION====\n', end='')
    test_nearest_neighbors(neural_editor, config)
    print('\n====STARTING NEURAL EDITOR EVALUATION====\n', end='')
    diffs_field = test_neural_editor_model(neural_editor, config)
    print('\n====STARTING COMMIT MSG GENERATOR EVALUATION====\n', end='')
    commit_msg_generator = torch.load(os.path.join(results_root, 'model_best_on_validation_commit_msg_generator.pt'),
                                      map_location=config['DEVICE'])
    print('\n====GREEDY====\n')
    test_commit_message_generation_model(commit_msg_generator, config, diffs_field, greedy=True)
    print('\n====BEAM SEARCH====\n')
    test_commit_message_generation_model(commit_msg_generator, config, diffs_field, greedy=False)


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

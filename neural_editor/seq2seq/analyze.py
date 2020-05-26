import os
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from torchtext.data import Field

from datasets.CodeChangesDataset import CodeChangesTokensDataset
from datasets.StablePatchPredictionDataset import StablePatchPredictionDataset
from datasets.dataset_utils import take_part_from_dataset
from neural_editor.seq2seq import EncoderDecoder, EncoderPredictor
from neural_editor.seq2seq.config import Config, load_config
from neural_editor.seq2seq.experiments.AccuracyCalculation import AccuracyCalculation
from neural_editor.seq2seq.experiments.BleuCalculation import BleuCalculation
from neural_editor.seq2seq.experiments.PredictorExamplesPrinting import PredictorExamplesPrinting
from neural_editor.seq2seq.experiments.PredictorMetricsCalculation import PredictorMetricsCalculation
from neural_editor.seq2seq.test_utils import save_predicted


def measure_experiment_time(func) -> Any:
    start = time.time()
    ret = func()
    end = time.time()
    print(f'Duration: {str(timedelta(seconds=end - start))}')
    print()
    return ret


def test_neural_editor_model(model: EncoderDecoder, config: Config) -> Field:
    train_dataset, val_dataset, test_dataset, diffs_field = \
        CodeChangesTokensDataset.load_data(verbose=True, config=config)
    if not config['USE_EDIT_REPRESENTATION']:
        print('Neural editor will not be tested because edit representations are not used.')
        return diffs_field
    train_dataset_test_size_part = take_part_from_dataset(train_dataset, len(test_dataset))
    accuracy_calculation_experiment = AccuracyCalculation(model, diffs_field, config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1,
                                                          greedy=False, config=config)
    bleu_calculation_experiment = BleuCalculation(config)

    model.eval()
    model.unset_edit_representation()
    with torch.no_grad():
        # Accuracy
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

    return diffs_field


def test_stable_patch_predictor_model(model: EncoderPredictor, diffs_field: Field, config: Config):
    train_dataset, val_dataset, test_dataset = \
        StablePatchPredictionDataset.load_data(diffs_field, verbose=True, config=config)
    train_dataset_test_size_part = take_part_from_dataset(train_dataset, len(test_dataset))
    predictor_metrics_calculation_experiment = PredictorMetricsCalculation(model, config)
    predictor_examples_printing_experiment = PredictorExamplesPrinting(model, n=5, config=config)

    model.eval()
    with torch.no_grad():
        # Printing examples
        measure_experiment_time(
            lambda: predictor_examples_printing_experiment.conduct(test_dataset, 'Test dataset')
        )
        measure_experiment_time(
            lambda: predictor_examples_printing_experiment.conduct(val_dataset, 'Validation dataset')
        )
        measure_experiment_time(
            lambda: predictor_examples_printing_experiment.conduct(
                train_dataset_test_size_part, f'Train dataset (test size approximation)')
        )

        # Metrics
        measure_experiment_time(
            lambda: predictor_metrics_calculation_experiment.conduct(test_dataset, 'Test dataset')
        )
        measure_experiment_time(
            lambda: predictor_metrics_calculation_experiment.conduct(val_dataset, 'Validation dataset')
        )
        measure_experiment_time(
            lambda: predictor_metrics_calculation_experiment.conduct(
                train_dataset_test_size_part, f'Train dataset (test size approximation)')
        )


def print_results(results_root: str, config: Config) -> None:
    pprint.pprint(config.get_config())
    neural_editor = None
    if config['USE_EDIT_REPRESENTATION']:
        neural_editor = torch.load(os.path.join(results_root, 'model_best_on_validation_neural_editor.pt'),
                                   map_location=config['DEVICE'])
    print('\n====STARTING NEURAL EDITOR EVALUATION====\n', end='')
    diffs_field = test_neural_editor_model(neural_editor, config)
    print('\n====STARTING STABLE PATCH PREDICTOR EVALUATION====\n', end='')
    stable_patch_predictor = torch.load(os.path.join(results_root, 'model_best_on_validation_predictor.pt'),
                                        map_location=config['DEVICE'])
    test_stable_patch_predictor_model(stable_patch_predictor, diffs_field, config)


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

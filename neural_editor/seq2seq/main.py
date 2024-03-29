"""
This file runs both training and evaluation
"""
import sys
from pathlib import Path

import torch

from datasets.CodeChangesDataset import CodeChangesTokensDataset
from datasets.StablePatchPredictionDataset import StablePatchPredictionDataset
from neural_editor.seq2seq.analyze import test_neural_editor_model, test_stable_patch_predictor_model
from neural_editor.seq2seq.config import load_config
from neural_editor.seq2seq.train import run_train
from neural_editor.seq2seq.train_predictor import run_train_predictor


def main():
    is_test = len(sys.argv) > 1 and sys.argv[1] == 'test'
    results_root = None if len(sys.argv) < 3 else Path(sys.argv[2])
    config = load_config(is_test, None if results_root is None else results_root.joinpath('config.pkl'))
    if results_root is not None and results_root.joinpath('model_checkpoint_neural_editor.pt').is_file():
        neural_editor = torch.load(str(results_root.joinpath('model_checkpoint_neural_editor.pt').absolute()),
                                   map_location=config['DEVICE'])
    else:
        neural_editor = None
    print('\n====STARTING TRAINING OF NEURAL EDITOR====\n', end='')
    train_dataset, val_dataset, test_dataset, diffs_field = \
        CodeChangesTokensDataset.load_data(config['VERBOSE'], config)
    neural_editor = run_train(train_dataset, val_dataset, diffs_field,
                              'neural_editor', config=config,
                              only_make_model=not config['USE_EDIT_REPRESENTATION'], model=neural_editor)
    print('\n====STARTING TRAINING OF STABLE PATCH PREDICTOR====\n', end='')
    if config['TRAIN_STABLE_PATCH_PREDICTOR']:
        train_dataset_stable_patches, val_dataset_stable_patches, test_dataset_stable_patches = \
            StablePatchPredictionDataset.load_data(diffs_field, config['VERBOSE'], config)
        stable_patch_predictor = run_train_predictor(train_dataset_stable_patches, val_dataset_stable_patches,
                                                     neural_editor, config=config)
        print('\n====STARTING STABLE PATCH PREDICTOR EVALUATION====\n', end='')
        test_stable_patch_predictor_model(stable_patch_predictor, diffs_field, config)
    else:
        print('NO STABLE PATCH PREDICTOR WILL BE TRAINED')
    print('\n====STARTING NEURAL EDITOR EVALUATION====\n', end='')
    test_neural_editor_model(neural_editor, config)


if __name__ == "__main__":
    main()

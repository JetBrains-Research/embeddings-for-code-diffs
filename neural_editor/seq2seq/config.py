import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.backends.cudnn


class Config:
    _CONFIG = {
        'IS_TEST': False,
        'DATASET_ROOT': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/0_50',
        'DATASET_ROOT_COMMIT': '../../../embeddings-for-code-diffs-data/datasets/stable_patches_detection/PatchNet_5_canonicalized_low_frequency_10_intersected/patchnet_timestamps/fold_1',
        'TUFANO_LABELED_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/0_50',
        'TUFANO_LABELED_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/50_100',
        'TRAIN_STABLE_PATCH_PREDICTOR': True,
        'FREEZE_EDIT_ENCODER_WEIGHTS': True,
        'TOKENS_CODE_CHUNK_MAX_LEN': 300,
        'CONTEXT_SIZE_FOR_HUNKS': 10,
        'OUTPUT_PATH': '../../../embeddings-for-code-diffs-data/last_execution/',
        'COMMIT_HASHES_PATH': None,
        'BLEU_PERL_SCRIPT_PATH': './experiments/multi-bleu.perl',  # Path to BLEU script calculator
        'UNK_TOKEN': "<unk>",
        'PAD_TOKEN': "<pad>",
        'SOS_TOKEN': "<s>",
        'EOS_TOKEN': "</s>",
        'LOWER': True,  # TODO: find out correlation between copying mechanism and lowering msg, it is tricky parameter
        'SEED': 9382,
        'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'TOKEN_MIN_FREQ': 1,
        'LEARNING_RATE': 0.0001,
        'MAX_NUM_OF_EPOCHS': 1000,
        'EDIT_REPRESENTATION_SIZE': 256,
        'WORD_EMBEDDING_SIZE': 128,
        'ENCODER_HIDDEN_SIZE': 128,
        'DECODER_HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'USE_EDIT_REPRESENTATION': True,
        'NUM_LAYERS_COMMIT': 2,
        'USE_COPYING_MECHANISM': False,
        'TEACHER_FORCING_RATIO': 0.9,
        'DROPOUT': 0.2,
        'USE_BRIDGE': True,
        'EARLY_STOPPING_ROUNDS': 10,
        'EVALUATION_ROUNDS_CLASSIFIER': 1000000,  # in batches
        'EARLY_STOPPING_ROUNDS_CLASSIFIER': 50,  # in evaluation rounds
        'BEAM_SIZE': 50,
        'NUM_GROUPS': 1,
        'DIVERSITY_STRENGTH': None,
        'TOP_K': [1, 3, 5, 10, 50],
        'BEST_ON': 'PPL',
        'BEST_ON_PREDICTOR': 'loss',  # TODO: implement
        'START_BEST_FROM_EPOCH': 0,
        'REPLACEMENT_TOKEN': 'замена',
        'DELETION_TOKEN': 'удаление',
        'ADDITION_TOKEN': 'добавление',
        'UNCHANGED_TOKEN': 'равенство',
        'PADDING_TOKEN': 'паддинг',
        'HUNK_TOKEN': 'кусок',
        'LEAVE_ONLY_CHANGED': True,
        'VERBOSE': True,
        'BATCH_SIZE': 64,
        'TSNE_BATCH_SIZE': 1024,
        'VAL_BATCH_SIZE': 64,
        'TEST_BATCH_SIZE': 64,
        'SAVE_MODEL_EVERY': 5,
        'PRINT_EVERY_iTH_BATCH': 5,
        'MAKE_CUDA_REPRODUCIBLE': False,
        'MAX_NUMBER_OF_EXAMPLES_TRAIN': None,
        'MAX_NUMBER_OF_EXAMPLES_VAL': None,
        'MAX_NUMBER_OF_EXAMPLES_TEST': None,
    }

    _PATH_KEYS = ['DATASET_ROOT', 'DATASET_ROOT_COMMIT', 'TUFANO_LABELED_0_50_PATH', 'TUFANO_LABELED_50_100_PATH',
                  'OUTPUT_PATH', 'BLEU_PERL_SCRIPT_PATH', 'COMMIT_HASHES_PATH']

    def __getitem__(self, key: str) -> Any:
        if key in self._PATH_KEYS and self._CONFIG[key] is not None:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), self._CONFIG[key]))
        return self._CONFIG[key]

    def save(self) -> None:
        with open(os.path.join(self['OUTPUT_PATH'], 'config.pkl'), 'wb') as config_file:
            pickle.dump(self._CONFIG, config_file)

    def get_config(self) -> Dict[str, Any]:
        return self._CONFIG.copy()

    def change_config_for_test(self) -> None:
        self._CONFIG['IS_TEST'] = True
        self._CONFIG['EVALUATION_ROUNDS_CLASSIFIER'] = 2
        self._CONFIG['MAX_NUM_OF_EPOCHS'] = 2
        self._CONFIG['BATCH_SIZE'] = 5
        self._CONFIG['VAL_BATCH_SIZE'] = 5
        self._CONFIG['TEST_BATCH_SIZE'] = 5
        self._CONFIG['MAX_NUMBER_OF_EXAMPLES_TRAIN'] = 64
        self._CONFIG['MAX_NUMBER_OF_EXAMPLES_VAL'] = 32
        self._CONFIG['MAX_NUMBER_OF_EXAMPLES_TEST'] = 32


def load_config(is_test: bool, path_to_config: Path = None) -> Config:
    config = Config()
    if path_to_config is not None:
        with path_to_config.open(mode='rb') as config_file:
            config._CONFIG = pickle.load(config_file)
            config._CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if config['SEED'] is not None:
        make_reproducible(config['SEED'], config['MAKE_CUDA_REPRODUCIBLE'])
    if is_test:
        config.change_config_for_test()
    return config


def make_reproducible(seed: int, make_cuda_reproducible: bool) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_cuda_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


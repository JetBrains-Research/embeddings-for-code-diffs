import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch.backends.cudnn


def get_dataset_path(dataset_suffix: str) -> str:
    return os.path.join(os.path.dirname(__file__), '../../../embeddings-for-code-diffs-data/datasets', dataset_suffix)


class Config:
    _CONFIG = {
        'IS_TEST': False,
        'DATASET_ROOT': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/0_50',
        'MATRIX_N_NEIGHBORS': 50,
        'TOKENS_CODE_CHUNK_MAX_LEN': 100,
        'METRIC': 'minkowski',
        'JIANG_WEIGHTS_LOADING': None,
        'LOAD_WEIGHTS_FROM': None,
        'DEFECTS4J_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/Defects4J',
        'TUFANO_LABELED_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/0_50',
        'TUFANO_LABELED_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/50_100',
        'OUTPUT_PATH': '../../../embeddings-for-code-diffs-data/last_execution/',
        'TUFANO_BUG_FIXES_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/0_50',
        'TUFANO_BUG_FIXES_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/50_100',
        'TUFANO_CODE_CHANGES_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/0_50',
        'TUFANO_CODE_CHANGES_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/50_100',
        'UNK_TOKEN': "<unk>",
        'PAD_TOKEN': "<pad>",
        'SOS_TOKEN': "<s>",
        'EOS_TOKEN': "</s>",
        'LOWER': False,
        'SEED': 9382,
        'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'TOKEN_MIN_FREQ': 1,
        'LEARNING_RATE': 0.0001,
        'MAX_NUM_OF_EPOCHS': 1000,
        'EDIT_REPRESENTATION_SIZE': 16,
        'WORD_EMBEDDING_SIZE': 128,
        'ENCODER_HIDDEN_SIZE': 128,
        'DECODER_HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'USE_EDIT_REPRESENTATION': True,
        'TEACHER_FORCING_RATIO': 0.9,
        'DROPOUT': 0.2,
        'USE_BRIDGE': True,
        'EARLY_STOPPING_ROUNDS': 25,
        'EARLY_STOPPING_ROUNDS_CLASSIFIER': 100,
        'EVALUATION_ROUNDS_CLASSIFIER': 100,
        'CLASSIFIER_EARLY_STOPPING_NO_EPOCHS': True,
        'CLASSIFIER_FILENAME': 'model_state_dict_best_on_validation_classifier.pt',
        'BEAM_SIZE': 50,
        'NUM_GROUPS': 1,
        'DIVERSITY_STRENGTH': None,
        'TOP_K': [1, 3, 5, 10, 50],
        'REPLACEMENT_TOKEN': 'замена',
        'DELETION_TOKEN': 'удаление',
        'ADDITION_TOKEN': 'добавление',
        'UNCHANGED_TOKEN': 'равенство',
        'PADDING_TOKEN': 'паддинг',
        'LEAVE_ONLY_CHANGED': True,
        'ADD_REVERSE_EXAMPLES_FOR_TRAIN_RATIO': 0.0,
        'BUILD_EDIT_VECTORS_EACH_QUERY': True,
        'UPDATE_TRAIN_VECTORS_EVERY_iTH_EPOCH': {'measure': 'epochs', 'period': 1},
        'LOSS_FUNCTION_PARAMS': {'measure': 'batches', 'default_loss_period': 1.0, 'bug_fixing_loss_period': 1.0},
        'VERBOSE': True,
        'BATCH_SIZE': 64,
        'TSNE_BATCH_SIZE': 1024,
        'VAL_BATCH_SIZE': 64,
        'TEST_BATCH_SIZE': 64,  # TODO: find out why changing batch size for dataloader changes perplexity
        'SAVE_MODEL_EVERY': 5,
        'PRINT_EVERY_iTH_BATCH': 5,
        'MAKE_CUDA_REPRODUCIBLE': False,
    }

    _PATH_KEYS = ['DATASET_ROOT', 'DEFECTS4J_PATH', 'LOAD_WEIGHTS_FROM',
                  'TUFANO_LABELED_0_50_PATH', 'TUFANO_LABELED_50_100_PATH',
                  'OUTPUT_PATH',
                  'TUFANO_BUG_FIXES_0_50_PATH', 'TUFANO_BUG_FIXES_50_100_PATH',
                  'TUFANO_CODE_CHANGES_0_50_PATH', 'TUFANO_CODE_CHANGES_50_100_PATH']

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
        self._CONFIG['DATASET_ROOT'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes_test/0_50'
        self._CONFIG['TUFANO_LABELED_0_50_PATH'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes_test/labeled/0_50'
        self._CONFIG['TUFANO_LABELED_50_100_PATH'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes_test/labeled/50_100'
        self._CONFIG['MAX_NUM_OF_EPOCHS'] = 2


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


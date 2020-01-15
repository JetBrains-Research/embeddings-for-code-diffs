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
        'DATASET_ROOT': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/0_50',
        'DEFECTS4J_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/Defects4J',
        'TUFANO_LABELED_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/0_50',
        'OUTPUT_PATH': '../../../embeddings-for-code-diffs-data/last_execution/',
        'UNK_TOKEN': "<unk>",
        'PAD_TOKEN': "<pad>",
        'SOS_TOKEN': "<s>",
        'EOS_TOKEN': "</s>",
        'LOWER': False,
        'SEED': 9382,
        'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'TOKENS_CODE_CHUNK_MAX_LEN': 50,
        'TOKEN_MIN_FREQ': 1,
        'LEARNING_RATE': 0.0001,
        'MAX_NUM_OF_EPOCHS': 1000,
        'EDIT_REPRESENTATION_SIZE': 16,
        'WORD_EMBEDDING_SIZE': 128,
        'ENCODER_HIDDEN_SIZE': 128,
        'DECODER_HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'DROPOUT': 0.2,
        'USE_BRIDGE': True,
        'EARLY_STOPPING_ROUNDS': 10,
        'BEAM_SIZE': 50,
        'NUM_GROUPS': 1,
        'DIVERSITY_STRENGTH': None,
        'TOP_K': [1, 3, 5, 10, 50],
        'REPLACEMENT_TOKEN': 'замена',
        'DELETION_TOKEN': 'удаление',
        'ADDITION_TOKEN': 'добавление',
        'UNCHANGED_TOKEN': 'равенство',
        'PADDING_TOKEN': 'паддинг',
        'VERBOSE': True,
        'BATCH_SIZE': 64,
        'TSNE_BATCH_SIZE': 1024,
        'VAL_BATCH_SIZE': 64,
        'TEST_BATCH_SIZE': 64,  # TODO: find out why changing batch size for dataloader changes perplexity
        'SAVE_MODEL_EVERY': 5,
        'PRINT_EVERY_iTH_BATCH': 5,
        'MAKE_CUDA_REPRODUCIBLE': False,
    }

    _PATH_KEYS = ['DATASET_ROOT', 'DEFECTS4J_PATH', 'TUFANO_LABELED_PATH', 'OUTPUT_PATH']

    def __getitem__(self, key: str) -> Any:
        if key in self._PATH_KEYS:
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
        self._CONFIG['TUFANO_LABELED_PATH'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes_test/labeled/0_50'
        self._CONFIG['MAX_NUM_OF_EPOCHS'] = 2


def load_config(is_test: bool, path_to_config: Path = None) -> Config:
    config = Config()
    if path_to_config is not None:
        with path_to_config.open(mode='rb') as config_file:
            config._CONFIG = pickle.load(config_file)
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


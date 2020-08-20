import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.backends.cudnn


class Config:
    _CONFIG = {
        'TRAIN_CMG': False,
        'IS_TEST': False,
        'IS_COMMIT_GENERATION': False,
        'DATASET_ROOT': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/0_50',
        'DATASET_ROOT_COMMIT': '../../../embeddings-for-code-diffs-data/datasets/commit_message_generation/Tufano',
        'FREEZE_EDIT_ENCODER_WEIGHTS': True,
        'TOKENS_CODE_CHUNK_MAX_LEN': 121,
        'MSG_MAX_LEN': 30,
        'DEFECTS4J_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/Defects4J',
        'TUFANO_LABELED_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/0_50',
        'TUFANO_LABELED_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/labeled/50_100',
        'OUTPUT_PATH': '../../../embeddings-for-code-diffs-data/last_execution/',
        'TUFANO_BUG_FIXES_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/0_50',
        'TUFANO_BUG_FIXES_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes/50_100',
        'TUFANO_CODE_CHANGES_0_50_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/0_50',
        'TUFANO_CODE_CHANGES_50_100_PATH': '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes/50_100',
        'BLEU_PERL_SCRIPT_PATH': './experiments/multi-bleu.perl',  # Path to BLEU script calculator
        'UNK_TOKEN': "<unk>",
        'PAD_TOKEN': "<pad>",
        'SOS_TOKEN': "<s>",
        'EOS_TOKEN': "</s>",
        'LOWER': True,  # TODO: find out correlation between copying mechanism and lowering msg, it is tricky parameter
        'LOWER_COMMIT_MSG': True,
        'SEED': 9382,
        'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'TOKEN_MIN_FREQ': 1,
        'LEARNING_RATE': 0.0001,
        'MAX_NUM_OF_EPOCHS': {'ne': 1000, 'cmg': 5000},
        'EDIT_REPRESENTATION_SIZE': 1,
        'WORD_EMBEDDING_SIZE': 128,
        'ENCODER_HIDDEN_SIZE': 128,
        'DECODER_HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'ANALYZE_NE': True,
        'TRAIN_NE': True,
        'USE_EDIT_REPRESENTATION': False,
        'NUM_LAYERS_COMMIT': 2,
        'USE_COPYING_MECHANISM': {'cmg': True, 'ne': False},
        'CONDUCT_EVALUATION_ON_TUFANO_AND_DEFECTS4J': False,
        'TEACHER_FORCING_RATIO': 0.9,
        'DROPOUT': 0.2,
        'USE_BRIDGE': True,
        'EARLY_STOPPING_ROUNDS': {'ne': 25, 'cmg': 100},
        'BEAM_SIZE': 5,
        'NUM_GROUPS': 1,
        'DIVERSITY_STRENGTH': None,
        'TOP_K': [1, 3, 5],
        'BEST_ON': {'ne': 'PPL', 'cmg': 'BLEU'},
        'START_BEST_FROM_EPOCH': 0,
        'REPLACEMENT_TOKEN': 'замена',
        'DELETION_TOKEN': 'удаление',
        'ADDITION_TOKEN': 'добавление',
        'UNCHANGED_TOKEN': 'равенство',
        'PADDING_TOKEN': 'паддинг',
        'LEAVE_ONLY_CHANGED': True,
        'VERBOSE': True,
        'BATCH_SIZE': 64,
        'TSNE_BATCH_SIZE': 1024,
        'VAL_BATCH_SIZE': 64,
        'TEST_BATCH_SIZE': 64,
        'SAVE_MODEL_EVERY': 5,
        'PRINT_EVERY_iTH_BATCH': 5,
        'MAKE_CUDA_REPRODUCIBLE': False,
    }

    _PATH_KEYS = ['DATASET_ROOT', 'DATASET_ROOT_COMMIT', 'DEFECTS4J_PATH',
                  'TUFANO_LABELED_0_50_PATH', 'TUFANO_LABELED_50_100_PATH',
                  'OUTPUT_PATH',
                  'TUFANO_BUG_FIXES_0_50_PATH', 'TUFANO_BUG_FIXES_50_100_PATH',
                  'TUFANO_CODE_CHANGES_0_50_PATH', 'TUFANO_CODE_CHANGES_50_100_PATH',
                  'BLEU_PERL_SCRIPT_PATH']

    def __getitem__(self, key: str) -> Any:
        if key in self._PATH_KEYS:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), self._CONFIG[key]))
        if isinstance(self._CONFIG[key], dict) and 'ne' in self._CONFIG[key] and 'cmg' in self._CONFIG[key]:
            return self._CONFIG[key]['cmg' if self._CONFIG['IS_COMMIT_GENERATION'] else 'ne']
        return self._CONFIG[key]

    def set_cmg_mode(self, is_cmg_mode):
        self._CONFIG['IS_COMMIT_GENERATION'] = is_cmg_mode

    def save(self) -> None:
        with open(os.path.join(self['OUTPUT_PATH'], 'config.pkl'), 'wb') as config_file:
            pickle.dump(self._CONFIG, config_file)

    def get_config(self) -> Dict[str, Any]:
        return self._CONFIG.copy()

    def change_config_for_test(self) -> None:
        self._CONFIG['IS_TEST'] = True
        self._CONFIG['DATASET_ROOT'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_bug_fixes_test/0_50'
        self._CONFIG['DATASET_ROOT_COMMIT'] = \
            '../../../embeddings-for-code-diffs-data/datasets/commit_message_generation/Jiang/filtered_dataset_test/partitioned/commit_message_generator'
        self._CONFIG['TUFANO_LABELED_0_50_PATH'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes_test/labeled/0_50'
        self._CONFIG['TUFANO_LABELED_50_100_PATH'] = \
            '../../../embeddings-for-code-diffs-data/datasets/java/tufano_code_changes_test/labeled/50_100'
        self._CONFIG['MAX_NUM_OF_EPOCHS'] = {'ne': 2, 'cmg': 3}


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


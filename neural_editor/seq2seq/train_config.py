import os
import random

import numpy as np
import torch.backends.cudnn


def make_reproducible(seed: int, make_cuda_reproducible: bool) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_cuda_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CONFIG = {
    'IS_TEST': False,
    'DATASET_ROOT': os.path.abspath(
        os.path.join(os.path.dirname(__file__), './data/datasets/java/tufano_bug_fixes/0_50')
    ),
    'OUTPUT_PATH': os.path.abspath(os.path.join(os.path.dirname(__file__), './data')),  # TODO: Fix this path for HSE cluster
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
    'EDIT_REPRESENTATION_SIZE': 512,
    'WORD_EMBEDDING_SIZE': 128,
    'ENCODER_HIDDEN_SIZE': 128,
    'DECODER_HIDDEN_SIZE': 256,
    'NUM_LAYERS': 2,
    'DROPOUT': 0.2,
    'USE_BRIDGE': True,
    'EARLY_STOPPING_ROUNDS': 1000,
    'BEAM_SIZE': 5,
    'REPLACEMENT_TOKEN': 'замена',
    'DELETION_TOKEN': 'удаление',
    'ADDITION_TOKEN': 'добавление',
    'UNCHANGED_TOKEN': 'равенство',
    'PADDING_TOKEN': 'паддинг',
    'VERBOSE': True,
    'BATCH_SIZE': 64,
    'VAL_BATCH_SIZE': 64,
    'TEST_BATCH_SIZE': 1,
    'SAVE_MODEL_EVERY': 10,
    'PRINT_EVERY_iTH_BATCH': 5,
    'MAKE_CUDA_REPRODUCIBLE': False,
}


def change_config_for_test():
    CONFIG['IS_TEST'] = True
    CONFIG['DATASET_ROOT'] = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                          "./data/datasets/java/tufano_bug_fixes_test/0_50"))
    CONFIG['MAX_NUM_OF_EPOCHS'] = 2


if CONFIG['SEED'] is not None:
    make_reproducible(CONFIG['SEED'], CONFIG['MAKE_CUDA_REPRODUCIBLE'])

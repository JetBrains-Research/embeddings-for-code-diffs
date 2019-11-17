import os

import numpy as np
import torch


def make_reproducible(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


CONFIG = {
    'DATASET_ROOT': os.path.abspath(os.path.join(os.path.dirname(__file__), "./.data/mined_diffs")),
    'UNK_TOKEN': "<unk>",
    'PAD_TOKEN': "<pad>",
    'SOS_TOKEN': "<s>",
    'EOS_TOKEN': "</s>",
    'LOWER': False,
    'SEED': 9382,
    'SPLIT_ON_TRAIN_VAL_TEST': False,
    'USE_CUDA': torch.cuda.is_available(),
    'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'TOKENS_CODE_CHUNK_MAX_LEN': 100,
    'TOKEN_MIN_FREQ': 1,
    'LEARNING_RATE': 0.0003,
    'MAX_NUM_OF_EPOCHS': 10,
    'CREATE_TOKEN_STRINGS': True,
    'EDIT_REPRESENTATION_SIZE': 512,
    'WORD_EMBEDDING_SIZE': 128,
    'ENCODER_HIDDEN_SIZE': 128,
    'DECODER_HIDDEN_SIZE': 256,
    'EARLY_STOPPING_ROUNDS': 1000,
    'BEAM_SIZE': 5,
    'REPLACEMENT_TOKEN': 'замена',
    'DELETION_TOKEN': 'удаление',
    'ADDITION_TOKEN': 'добавление',
    'UNCHANGED_TOKEN': 'равенство',
    'PADDING_TOKEN': 'паддинг',
    'VERBOSE': True,
    'PRINT_EVERY_EPOCH': 5,
}

if CONFIG['SEED'] is not None:
    make_reproducible(CONFIG['SEED'])

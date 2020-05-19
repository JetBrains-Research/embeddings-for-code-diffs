import torch
from torch import nn

from edit_representation.sequence_encoding.EditEncoder import EditEncoder
from neural_editor.seq2seq.classifier import GoodEditClassifier
from neural_editor.seq2seq.classifier.GoodEditClassifier import GoodEditClassifier
from neural_editor.seq2seq.config import Config


def load_classifier(state_dict_path: str, vocab_size, config) -> GoodEditClassifier:
    model = create_classifier(vocab_size, config)
    model.load_state_dict(torch.load(state_dict_path, map_location=config['DEVICE']))
    return model


def create_classifier(vocab_size, config: Config):
    emb_size = config['WORD_EMBEDDING_SIZE']
    embedding = nn.Embedding(vocab_size, emb_size)
    original_src_encoder = EditEncoder(emb_size, config['ENCODER_HIDDEN_SIZE'], config['NUM_LAYERS'], config['DROPOUT'])
    edit_src_encoder = original_src_encoder
    model = GoodEditClassifier(original_src_encoder, edit_src_encoder, embedding, output_size=1)
    model.to(config['DEVICE'])
    return model
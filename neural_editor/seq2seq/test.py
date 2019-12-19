from typing import List

import numpy as np
import torch
from torchtext import data
from torchtext.data import Dataset, Field
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.test_utils import visualize_tsne
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import print_examples, rebatch, calculate_accuracy


def visualization_for_classified_dataset(model: EncoderDecoder, dataset: Dataset, classes: List[str], diffs_field: Field) -> None:
    pad_index: int = diffs_field.vocab.stoi[CONFIG['PAD_TOKEN']]
    iterator = data.Iterator(dataset, batch_size=1,
                             sort=False, train=False, shuffle=False, device=CONFIG['DEVICE'])
    representations = torch.zeros(len(dataset), CONFIG['EDIT_REPRESENTATION_SIZE'] * 2)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch = rebatch(pad_index, batch)
            representations[i: i + 1] = model.encode_edit(batch)[0][-1, :]  # hidden, last layer, all batches
        visualize_tsne(representations, classes)


def visualization_for_unclassified_dataset(model: EncoderDecoder, dataset: Dataset, diffs_field: Field) -> None:
    pad_index: int = diffs_field.vocab.stoi[CONFIG['PAD_TOKEN']]
    batch_size = len(dataset)

    def batch_to_comparable_element(x):
        return len(x.src), len(x.trg)

    iterator = data.Iterator(dataset, batch_size=batch_size,
                             sort=False, train=False,
                             sort_within_batch=True,
                             sort_key=batch_to_comparable_element, shuffle=False, device=CONFIG['DEVICE'])
    representations = torch.zeros(len(dataset), CONFIG['EDIT_REPRESENTATION_SIZE'] * 2)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch = rebatch(pad_index, batch)
            representations[i: i + batch_size] = model.encode_edit(batch)[0][-1, :]  # hidden, last layer, all batches
        visualize_tsne(representations, None)


def one_shot_learning(model: EncoderDecoder, dataset: Dataset, classes: List[str], diffs_field: Field) -> None:
    vocab: Vocab = diffs_field.vocab
    pad_index: int = vocab.stoi[CONFIG['PAD_TOKEN']]
    iter = data.Iterator(dataset, batch_size=1, sort=False, train=False, shuffle=False, device=CONFIG['DEVICE'])
    # noinspection PyTypeChecker
    # reason: None is not a type of Optimizer
    current_class = None
    correct_same_edit_representation = 0
    total_same_edit_representation = 0
    correct_other_edit_representation = 0
    total_other_edit_representation = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iter):
            batch = rebatch(pad_index, batch)
            y = classes[i]
            if y != current_class:
                print(f'NEW CLASS: {y}')
                model.set_edit_representation(batch)
                current_class = y
                correct = calculate_accuracy([batch], model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'], vocab)
                correct_same_edit_representation += correct
                total_same_edit_representation += 1
            else:
                correct = calculate_accuracy([batch], model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'], vocab)
                correct_other_edit_representation += correct
                total_other_edit_representation += 1
            if correct == 1:
                color = 'green'
            else:
                color = 'red'
            print_examples([batch], model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'], vocab, n=1, color=color)
        print(f'Accuracy on Defects4J for same edit representations: '
              f'{correct_same_edit_representation} / {total_same_edit_representation} = '
              f'{correct_same_edit_representation / total_same_edit_representation}')
        print(f'Accuracy on Defects4J for other edit representations: '
              f'{correct_other_edit_representation} / {total_other_edit_representation} = '
              f'{correct_other_edit_representation / total_other_edit_representation}')
        model.unset_edit_representation()

from typing import List

import torch
from torchtext import data
from torchtext.data import Dataset, Field

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.test_utils import visualize_tsne
from neural_editor.seq2seq.train_utils import rebatch


def visualization_for_classified_dataset(model: EncoderDecoder, dataset: Dataset, classes: List[str],
                                         diffs_field: Field, config: Config) -> None:
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    iterator = data.Iterator(dataset, batch_size=1,
                             sort=False, train=False, shuffle=False, device=config['DEVICE'])
    representations = torch.zeros(len(dataset), config['EDIT_REPRESENTATION_SIZE'] * 2)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch = rebatch(pad_index, batch, config)
            representations[i: i + 1] = model.encode_edit(batch)[0][-1, :]  # hidden, last layer, all batches
        visualize_tsne(representations, classes, config)


def visualization_for_unclassified_dataset(model: EncoderDecoder, dataset: Dataset,
                                           diffs_field: Field, config: Config) -> None:
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    batch_size = len(dataset)

    def batch_to_comparable_element(x):
        return len(x.src), len(x.trg)

    iterator = data.Iterator(dataset, batch_size=batch_size,
                             sort=False, train=False,
                             sort_within_batch=True,
                             sort_key=batch_to_comparable_element, shuffle=False, device=config['DEVICE'])
    representations = torch.zeros(len(dataset), config['EDIT_REPRESENTATION_SIZE'] * 2)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch = rebatch(pad_index, batch, config)
            representations[i: i + batch_size] = model.encode_edit(batch)[0][-1, :]  # hidden, last layer, all batches
        visualize_tsne(representations, None, config)


from typing import List

import torch
from torchtext import data
from torchtext.data import Field, Dataset

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.test_utils import visualize_tsne
from neural_editor.seq2seq.train_utils import rebatch


class EditRepresentationVisualization:

    def __init__(self, model: EncoderDecoder, diffs_field: Field, config: Config) -> None:
        super().__init__()
        self.model = model
        self.diffs_field = diffs_field
        self.config = config
        self.pad_index: int = diffs_field.vocab.stoi[self.config['PAD_TOKEN']]

    def conduct(self, dataset: Dataset, filename: str, classes: List[str]):
        print(f'Starting conducting edit representation visualization experiment for {filename}...')
        if classes is None:
            self.visualization_for_unclassified_dataset(dataset, filename)
        else:
            self.visualization_for_classified_dataset(dataset, filename, classes)

    def visualization_for_classified_dataset(self, dataset: Dataset, filename: str, classes: List[str]) -> None:
        iterator = data.Iterator(dataset, batch_size=1,
                                 sort=False, train=False, shuffle=False, device=self.config['DEVICE'])
        representations = self.get_representations(iterator, len(dataset))
        visualize_tsne(representations, classes, filename, self.config)

    def visualization_for_unclassified_dataset(self, dataset: Dataset, filename: str) -> None:
        def batch_to_comparable_element(x):
            return len(x.src), len(x.trg)

        iterator = data.Iterator(dataset, batch_size=self.config['TSNE_BATCH_SIZE'],
                                 sort=False, train=False,
                                 sort_within_batch=True,
                                 sort_key=batch_to_comparable_element, shuffle=False, device=self.config['DEVICE'])

        representations = self.get_representations(iterator, len(dataset))
        visualize_tsne(representations, None, filename, self.config)

    def get_representations(self, iterator, dataset_size) -> torch.Tensor:
        representations = torch.zeros(dataset_size, self.config['EDIT_REPRESENTATION_SIZE'] * 2)
        cur_pos = 0
        for batch in iterator:
            batch = rebatch(self.pad_index, batch, self.config)
            representations[cur_pos: cur_pos + len(batch)] = self.model.encode_edit(batch)[0][-1, :]  # hidden, last layer, all batches
            cur_pos += len(batch)
        return representations

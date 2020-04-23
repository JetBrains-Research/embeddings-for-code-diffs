from typing import List, Callable, Dict, Optional

import torch
from sklearn.neighbors import NearestNeighbors
from torchtext import data
from torchtext.data import Dataset
import numpy as np

from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.Batch import rebatch
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.experiments.SimpleNearestNeighbors import SimpleNearestNeighbors


class NearestNeighbor:
    def __init__(self, model: EncoderDecoder, pad_index: int, config: Config) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.pad_index = pad_index
        self.differ = Differ(self.config['REPLACEMENT_TOKEN'], self.config['DELETION_TOKEN'],
                             self.config['ADDITION_TOKEN'], self.config['UNCHANGED_TOKEN'],
                             self.config['PADDING_TOKEN'])
        self.levenshtein_metric = self.create_levenshtein_metric()

    def create_levenshtein_metric(self) -> Callable:
        def levenshtein_metric(x: np.ndarray, y: np.ndarray) -> float:
            diff = self.differ.diff_tokens_fast_lvn(x, y, leave_only_changed=True)
            return len(diff[0])

        return levenshtein_metric

    def conduct(self, dataset_train: Dataset, dataset_test: Optional[Dataset], dataset_label: str) -> Dict[
        str, Dict[str, List[int]]]:
        print(f'Start conducting nearest neighbor experiment for {dataset_label}...')
        nbrs_result = self.find(dataset_train, dataset_test)
        return nbrs_result

    def find(self, dataset_train: Dataset, dataset_test: Optional[Dataset]) -> Dict[str, Dict[str, List[int]]]:
        nbrs = self.create_nearest_neighbors(dataset_train)
        encoded_data = self.encode_data(dataset_test)
        nbrs_result = self.get_metrics_dict()
        for vector_type in nbrs_result:
            for distance_type in nbrs_result[vector_type]:
                indices = nbrs[vector_type][distance_type].kneighbors(encoded_data[vector_type],
                                                                      return_distance=False)
                indices = indices[:, 0]
                nbrs_result[vector_type][distance_type] = indices
        return nbrs_result

    def get_metrics_dict(self):
        return {'src': {'levenshtein': self.levenshtein_metric},
                'src_hidden': {'minkowski': 'minkowski', 'cosine': 'cosine'},
                'edit_hidden': {'minkowski': 'minkowski', 'cosine': 'cosine'}}

    def create_nearest_neighbors(self, dataset_train):
        encoded_train = self.encode_data(dataset_train)
        nbrs = self.get_metrics_dict()
        for vector_type in nbrs:
            for distance_type in nbrs[vector_type]:
                metric = nbrs[vector_type][distance_type]
                if distance_type == 'levenshtein':
                    nbrs[vector_type][distance_type] = \
                        SimpleNearestNeighbors(metric=metric).fit(encoded_train[vector_type])
                else:
                    nbrs[vector_type][distance_type] = \
                        NearestNeighbors(n_neighbors=1, algorithm='brute', metric=metric, n_jobs=-1) \
                            .fit(encoded_train[vector_type])
        return nbrs

    def encode_data(self, dataset: Dataset):
        if dataset is None:
            return {'src': None, 'src_hidden': None, 'edit_hidden': None}

        ids = []
        encoded_data = {'src': [], 'src_hidden': [], 'edit_hidden': []}
        # TODO: batch != 1 produces different results
        data_iterator = data.Iterator(dataset, batch_size=self.config['BATCH_SIZE'], train=False,
                                      sort=False,
                                      sort_within_batch=True,
                                      sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                      device=self.config['DEVICE'])
        data_iterator = [rebatch(self.pad_index, batch, self.config) for batch in data_iterator]

        for batch in data_iterator:
            (edit_hidden, edit_cell), _, (encoder_hidden, _) = self.model.encode(batch, ignore_encoded_train=True)
            encoded_data['src_hidden'].append(encoder_hidden[-1].detach().cpu())
            encoded_data['edit_hidden'].append(edit_hidden[-1].detach().cpu())
            ids.append(batch.ids.detach().cpu())
        encoded_data['src_hidden'] = torch.cat(encoded_data['src_hidden'], dim=0)
        encoded_data['edit_hidden'] = torch.cat(encoded_data['edit_hidden'], dim=0)
        ids = torch.cat(ids, dim=0)
        ids_reverse = torch.argsort(ids)

        encoded_data_sorted = {'src': [example.src for example in dataset.examples],
                               'src_hidden': encoded_data['src_hidden'][ids_reverse, :].numpy(),
                               'edit_hidden': encoded_data['edit_hidden'][ids_reverse, :].numpy()}

        return encoded_data_sorted

from enum import Enum
from typing import List, Tuple, Dict

import sklearn
import torch
from torch import Tensor
from torchtext import data
from torchtext.data import Dataset, Field
import numpy as np
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.experiments.BleuCalculation import BleuCalculation
from neural_editor.seq2seq.test_utils import save_predicted
from neural_editor.seq2seq.train_utils import rebatch


class FeaturesType(Enum):
    ONLY_SRC = 1
    ONLY_EDIT = 2
    SRC_AND_EDIT = 3


def sequence_equality(x: np.ndarray, y: np.ndarray) -> float:
    return (np.abs(x - y) > 1e-10).sum()


class NearestNeighbors:
    def __init__(self, model: EncoderDecoder, target_field: Field, config: Config) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.trg_vocab: Vocab = target_field.vocab
        self.pad_index: int = self.trg_vocab.stoi[config['PAD_TOKEN']]
        self.bleu_calculation_experiment = BleuCalculation(config)

    def conduct(self, train_datasets: List[Dataset], test_datasets: Dict[str, Dataset],
                features_type: FeaturesType, training_label: str) -> None:
        print(f'Conducting Nearest Neighbors experiment with training on {training_label}')
        train_X, train_Y = self.extract_features(train_datasets, features_type)
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='brute', metric=sequence_equality).fit(train_X)
        for dataset_name, dataset in test_datasets.items():
            print(f'Evalution on {dataset_name}...')
            test_X, test_Y = self.extract_features([dataset], features_type)
            distances, indices = nbrs.kneighbors(test_X)
            prediction_tokens = [train_Y[idx] for idx in indices[:, 0]]
            prediction = [' '.join(pred) for pred in prediction_tokens]
            targets = [' '.join(msg) for msg in test_Y]

            correct, total = self.get_accuracy(prediction, targets)
            print(f'Accuracy: {correct} / {total} = {correct / total}')
            bleu_result = self.bleu_calculation_experiment.run_script(prediction, targets)
            print(bleu_result[0])
            print(f'Errors: {bleu_result[1]}')
            print('====')
            save_predicted([[pred] for pred in prediction_tokens],
                           dataset_name=f'nn_{dataset_name}_{training_label}',
                           config=self.config)
        print('END OF EXPERIMENT')

    def get_accuracy(self, prediction: List[str], targets: List[str]) -> Tuple[float, float]:
        prediction = np.array(prediction)
        targets = np.array(targets)
        return (prediction == targets).sum(), len(prediction)

    def extract_features_src(self, datasets: List[Dataset], features_type: FeaturesType) -> Tuple[np.ndarray, List[List[str]]]:
        features = []
        targets = []
        for dataset in datasets:
            targets += [example.msg for example in dataset.examples]
            dataset_features = None
            data_iterator = data.Iterator(dataset, batch_size=len(dataset), train=False,
                                          shuffle=False,
                                          sort=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          device=self.config['DEVICE'])
            data_iterator = (rebatch(self.pad_index, batch, dataset, self.config) for batch in data_iterator)
            for batch in data_iterator:
                batch_features = batch.src.detach().numpy()
                if dataset_features is None:
                    dataset_features = np.empty((len(dataset), self.config['TOKENS_CODE_CHUNK_MAX_LEN'])).astype(batch_features.dtype)
                    dataset_features.fill(self.pad_index)
                dataset_features[batch.ids, :batch_features.shape[1]] = batch_features
            features.append(dataset_features)
        return np.concatenate(features, axis=0), targets

    def extract_features(self, datasets: List[Dataset], features_type: FeaturesType) -> Tuple[np.ndarray, List[List[str]]]:
        features = []
        targets = []
        for dataset in datasets:
            targets += [example.msg for example in dataset.examples]
            dataset_features = None
            data_iterator = data.Iterator(dataset, batch_size=64, train=False,
                                          shuffle=False,
                                          sort=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          device=self.config['DEVICE'])
            data_iterator = (rebatch(self.pad_index, batch, dataset, self.config) for batch in data_iterator)
            for batch in data_iterator:
                edit_final, _, encoder_final = self.model.encode(batch)
                edit_final, encoder_final = edit_final[0][-1], encoder_final[0][-1]
                batch_features = self.compose_features(edit_final, encoder_final, features_type)
                if dataset_features is None:
                    dataset_features = np.empty((len(dataset), batch_features.shape[1])).astype(batch_features.dtype)
                dataset_features[batch.ids] = batch_features
            features.append(dataset_features)
        return np.concatenate(features, axis=0), targets

    def compose_features(self, edit: Tensor, src: Tensor, features_type: FeaturesType) -> np.ndarray:
        if features_type is FeaturesType.ONLY_SRC:
            return src.detach().numpy()
        elif features_type is FeaturesType.ONLY_EDIT:
            return edit.detach().numpy()
        elif features_type is FeaturesType.SRC_AND_EDIT:
            return torch.cat((src, edit), dim=-1).detach().numpy()
        else:
            raise AttributeError(f'No feature type {features_type}')

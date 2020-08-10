from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score
from torchtext import data
from torchtext.data import Dataset

from neural_editor.seq2seq import EncoderPredictor
from neural_editor.seq2seq.PredictorBatch import rebatch_predictor
from neural_editor.seq2seq.config import Config


def calculate_metrics(y_true, y_pred_labels, y_pred_probs) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels),
        'recall': recall_score(y_true, y_pred_labels),
        'f1_score': f1_score(y_true, y_pred_labels),
        'auc': roc_auc_score(y_true, y_pred_probs),
        'average_precision': average_precision_score(y_true, y_pred_probs)
    }
    return metrics


def concat_predictions(model, y_true: List[Tensor], y_pred_probs: List[Tensor]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred_probs = torch.cat(y_true), torch.cat(y_pred_probs)
    y_pred_labels = model.predict_labels(y_pred_probs)
    y_true, y_pred_labels, y_pred_probs = \
        y_true.detach().cpu().numpy(), y_pred_labels.detach().cpu().numpy(), y_pred_probs.detach().cpu().numpy()
    return y_true, y_pred_labels, y_pred_probs


class PredictorMetricsCalculation:
    def __init__(self, model: EncoderPredictor, config: Config) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def conduct(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting predictor metrics calculation experiment for {dataset_label}...', flush=True)
        data_iter = data.Iterator(dataset, batch_size=self.config['VAL_BATCH_SIZE'], train=False,
                                  sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False,
                                  device=self.config['DEVICE'])
        data_iter = [rebatch_predictor(b, dataset, self.config) for b in data_iter]
        y_true, y_pred_probs = [], []
        for batch in data_iter:
            batch_pred_probs = self.model.predict(batch)
            y_pred_probs.append(batch_pred_probs)
            y_true.append(batch.trg)
        y_true, y_pred_labels, y_pred_probs = concat_predictions(self.model, y_true, y_pred_probs)
        metrics = calculate_metrics(y_true, y_pred_labels, y_pred_probs)
        for name, value in metrics.items():
            print(f'{name}: {round(value, 3)}')


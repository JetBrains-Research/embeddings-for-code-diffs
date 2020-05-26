import time
from datetime import timedelta
from typing import List, Dict, Optional

import torch
from torch import nn
from torchtext import data

from neural_editor.seq2seq import EncoderPredictor
from neural_editor.seq2seq.PredictorBatch import rebatch_predictor
from neural_editor.seq2seq.PredictorLossCompute import PredictorLossCompute
from neural_editor.seq2seq.experiments.PredictorMetricsCalculation import calculate_metrics, concat_predictions
from neural_editor.seq2seq.test_utils import save_metrics_plot
from neural_editor.seq2seq.train_utils import make_predictor, load_weights_of_best_model_on_validation, \
    save_model


def aggregate_metrics(model: EncoderPredictor, total_metrics_dict: Optional[Dict[str, List]],
                      y_true: List[torch.Tensor], y_pred_probs: List[torch.Tensor],
                      loss: float) -> Dict[str, List]:
    y_true, y_pred_labels, y_pred_probs = concat_predictions(model, y_true, y_pred_probs)
    current_metrics = calculate_metrics(y_true, y_pred_labels, y_pred_probs)
    current_metrics['loss'] = loss
    return add_metrics(total_metrics_dict, current_metrics)


def add_metrics(total_metrics_dict: Optional[Dict[str, List]], current_metrics_dict: Dict[str, float]) -> Dict[str, List]:
    if total_metrics_dict is None:
        total_metrics_dict = {k: [] for k in current_metrics_dict}
    for k in total_metrics_dict:
        total_metrics_dict[k].append(current_metrics_dict[k])
    return total_metrics_dict


def validate(model, val_iter, val_loss_function, val_metrics):
    model.eval()
    with torch.no_grad():
        total_loss, total_nseqs, y_true, y_pred_probs = 0, 0, [], []
        for i, batch in enumerate(val_iter, 1):
            out = model.forward(batch)
            loss = val_loss_function(out, batch.trg)
            total_loss += loss * batch.nseqs
            total_nseqs += batch.nseqs
            probs = model.predict_from_forward(out)
            y_true.append(batch.trg)
            y_pred_probs.append(probs)
        return aggregate_metrics(model, val_metrics, y_true, y_pred_probs, total_loss / total_nseqs)


def train_classifier(model, train_dataset, val_dataset, config):
    epochs_num = config['MAX_NUM_OF_EPOCHS']
    early_stopping_rounds = config['EARLY_STOPPING_ROUNDS_CLASSIFIER']
    evaluation_rounds = config['EVALUATION_ROUNDS_CLASSIFIER']
    min_val_loss: float = 1000000
    num_not_decreasing_steps: int = 0

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['LEARNING_RATE'])
    train_metrics = None
    val_metrics = None
    train_loss_function = PredictorLossCompute(criterion, optimizer)
    val_loss_function = PredictorLossCompute(criterion, None)
    train_iter = data.BucketIterator(train_dataset, batch_size=config['BATCH_SIZE'], train=True,
                                     sort_within_batch=True, sort_key=lambda x: len(x.src),
                                     shuffle=True, repeat=False, device=config['DEVICE'])
    val_iter = data.Iterator(val_dataset, batch_size=config['VAL_BATCH_SIZE'], train=False,
                             sort_within_batch=True, sort_key=lambda x: len(x.src), repeat=False,
                             device=config['DEVICE'])

    epoch_start = time.time()
    start = time.time()
    for epoch in range(epochs_num):
        print(f'Epoch {epoch} / {epochs_num}')

        total_loss, total_nseqs, y_true, y_pred_probs = 0, 0, [], []
        batched_train_iter = [rebatch_predictor(b) for b in train_iter]
        for i, batch in enumerate(batched_train_iter, 1):
            if num_not_decreasing_steps == early_stopping_rounds:
                print(f'Training was early stopped on epoch {epoch} with early stopping rounds {early_stopping_rounds}')
                return train_metrics, val_metrics

            model.train()
            out = model.forward(batch)
            loss = train_loss_function(out, batch.trg)
            total_loss += loss * batch.nseqs
            total_nseqs += batch.nseqs
            probs = model.predict_from_forward(out)
            y_true.append(batch.trg)
            y_pred_probs.append(probs)

            if i % evaluation_rounds == 0 or i == len(batched_train_iter):
                train_metrics = aggregate_metrics(model, train_metrics, y_true, y_pred_probs, total_loss / total_nseqs)
                total_loss, total_nseqs, y_true, y_pred_probs = 0, 0, [], []

                val_metrics = validate(model, [rebatch_predictor(b) for b in val_iter], val_loss_function, val_metrics)
                if val_metrics['loss'][-1] < min_val_loss:
                    save_model(model, 'best_on_validation_predictor', config)
                    min_val_loss = val_metrics['loss'][-1]
                    num_not_decreasing_steps = 0
                else:
                    num_not_decreasing_steps += 1

                train_output = 'Train: ' + ', '.join([f'{metric_name} {round(train_metrics[metric_name][-1], 4)}'
                                                      for metric_name in train_metrics.keys()])
                val_output = 'Valid: ' + ', '.join([f'{metric_name} {round(val_metrics[metric_name][-1], 4)}'
                                                    for metric_name in val_metrics.keys()])
                elapsed = time.time() - start
                print(f'Epoch Step: {i} / {len(train_iter)} Elapsed time: {round(elapsed, 4)}')
                print(train_output)
                print(val_output)
                start = time.time()

        epoch_duration = time.time() - epoch_start
        print(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
        epoch_start = time.time()

    return train_metrics, val_metrics


def run_train_predictor(train_dataset, val_dataset, neural_editor, config):
    suffix = 'predictor'
    model = make_predictor(neural_editor.edit_encoder, neural_editor.encoder, config)
    train_metrics, val_metrics = train_classifier(model, train_dataset, val_dataset, config)
    print(f'Train: {train_metrics}')
    print(f'Validation: {val_metrics}')
    for key in train_metrics.keys():
        save_metrics_plot([train_metrics[key], val_metrics[key]], ['train', 'val'], key,
                          f'{key}_{suffix}.png', config, xlabel='iteration', ylabel=key)
    load_weights_of_best_model_on_validation(model, suffix, config)
    return model

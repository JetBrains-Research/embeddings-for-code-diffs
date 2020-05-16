import sys
import time
from typing import List, Tuple
from datetime import timedelta

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchtext import data

from edit_representation.sequence_encoding.EditEncoder import EditEncoder
from neural_editor.seq2seq.ClassifierBatch import rebatch_classifier, rebatch_classifier_iterator
from neural_editor.seq2seq.ClassifierLossCompute import ClassifierLossCompute
from neural_editor.seq2seq.analyze import load_all
from neural_editor.seq2seq.classifier.GoodEditClassifier import GoodEditClassifier
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.datasets.dataset_utils import create_classifier_dataset, take_part_from_dataset
from neural_editor.seq2seq.matrix_calculation import load_matrix, print_matrix_statistics, get_matrix, write_matrix
from neural_editor.seq2seq.test_utils import save_perplexity_plot
from neural_editor.seq2seq.train import save_model, load_weights_of_best_model_on_validation
from neural_editor.seq2seq.train_utils import lookup_words


def create_classifier(vocab_size, config: Config):
    emb_size = config['WORD_EMBEDDING_SIZE']
    embedding = nn.Embedding(vocab_size, emb_size)
    original_src_encoder = EditEncoder(emb_size, config['ENCODER_HIDDEN_SIZE'], config['NUM_LAYERS'], config['DROPOUT'])
    edit_src_encoder = original_src_encoder
    model = GoodEditClassifier(original_src_encoder, edit_src_encoder, embedding, output_size=1)
    model.to(config['DEVICE'])
    return model


def run_classifier_epoch(data_iter: List, model: GoodEditClassifier, loss_compute,
                         print_every: int) -> Tuple[float, float]:
    epoch_start = time.time()
    start = time.time()
    total_correct = 0
    total_loss = 0
    total_nseqs = 0
    for i, batch in enumerate(data_iter, 1):
        out = model.forward(batch)
        loss = loss_compute(out, batch.trg)
        total_loss += loss * out.shape[0]
        total_nseqs += batch.nseqs
        total_correct += (torch.round(torch.sigmoid(out)) == batch.trg).sum().data.item()
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print(f'Epoch Step: {i} / {len(data_iter)} '
                  f'Loss: {loss} '
                  f'Elapsed time: {elapsed}')
            start = time.time()
    epoch_duration = time.time() - epoch_start
    print(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
    return total_loss / total_nseqs, total_correct / total_nseqs


def classifier_metrics(model, dataset: Dataset, diffs_field, config, label=None):
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    data_iter = rebatch_classifier_iterator(data.Iterator(dataset, batch_size=config['BATCH_SIZE'], train=False,
                                                          sort=False,
                                                          sort_within_batch=False, repeat=False, shuffle=False,
                                                          device=config['DEVICE']), pad_index)
    tp, fp, tn, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            out = model.predict(batch)
            out = torch.round(out)
            tp += (out[batch.trg == 1] == 1).sum().data.item()
            fp += (out[batch.trg == 0] == 1).sum().data.item()
            tn += (out[batch.trg == 0] == 0).sum().data.item()
            fn += (out[batch.trg == 1] == 0).sum().data.item()
    if label is not None:
        accuracy = (tp + tn) / len(dataset)
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(f'\nMetrics on {label}')
        print(f'Accuracy: {tp + tn} / {len(dataset)} = {accuracy}')
        print(f'Precision: {tp} / {tp + fp} = {precision}')
        print(f'Recall: {tp} / {tp + fn} = {recall}')
        print(f'F1 score: {f1_score}')


def train_classifier(model, train_dataset, val_dataset, diffs_field, config):
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    epochs_num = config['MAX_NUM_OF_EPOCHS']
    early_stopping_rounds = config['EARLY_STOPPING_ROUNDS_CLASSIFIER']
    min_val_loss: float = 1000000
    num_not_decreasing_steps: int = 0

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['LEARNING_RATE'])
    losses = {'train': [], 'validation': []}
    accuracies = {'train': [], 'validation': []}
    train_loss_function = ClassifierLossCompute(criterion, optimizer)
    val_loss_function = ClassifierLossCompute(criterion, None)
    train_iter = data.Iterator(train_dataset, batch_size=config['BATCH_SIZE'], train=True,
                               sort=False, sort_within_batch=False, repeat=False, shuffle=True,
                               device=config['DEVICE'])
    val_iter = data.Iterator(val_dataset, batch_size=config['VAL_BATCH_SIZE'], train=False,
                             sort=False, sort_within_batch=False, repeat=False,
                             device=config['DEVICE'])

    for epoch in range(epochs_num):
        if num_not_decreasing_steps == early_stopping_rounds:
            print(f'Training was early stopped on epoch {epoch} with early stopping rounds {early_stopping_rounds}')
            break

        print(f'Epoch {epoch} / {epochs_num}')
        model.train()

        train_loss, train_acc = run_classifier_epoch([rebatch_classifier(pad_index, b) for b in train_iter],
                                                     model, train_loss_function,
                                                     print_every=config['PRINT_EVERY_iTH_BATCH'])
        print(f'Train loss: {train_loss}, accuracy: {train_acc}')
        losses['train'].append(train_loss)
        accuracies['train'].append(train_acc)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = run_classifier_epoch([rebatch_classifier(pad_index, t) for t in val_iter],
                                                          model, val_loss_function,
                                                          print_every=config['PRINT_EVERY_iTH_BATCH'])
            print(f'Validation loss: {val_loss}, accuracy: {val_accuracy}')
            losses['validation'].append(val_loss)
            accuracies['validation'].append(val_accuracy)
            if val_loss < min_val_loss:
                save_model(model, 'best_on_validation_classifier', config)
                min_val_loss = val_loss
                num_not_decreasing_steps = 0
            else:
                num_not_decreasing_steps += 1
    return losses, accuracies


def run_train_classifier(dataset, diffs_field, matrix, diff_example_ids, config):
    train_dataset, val_dataset, test_dataset = \
        create_classifier_dataset(dataset, diffs_field, matrix, diff_example_ids)
    print(f'\n====TRAINING CLASSIFIER====\n')
    print(f'Dataset sizes train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
    model = create_classifier(len(diffs_field.vocab), config)
    losses, accuracies = train_classifier(model, train_dataset, val_dataset, diffs_field, config)
    print(f'Losses: {losses}')
    print(f'Accuracies: {accuracies}')
    save_perplexity_plot(losses.values(), losses.keys(), 'loss_classifier.png', config)
    save_perplexity_plot(accuracies.values(), accuracies.keys(), 'accuracy_classifier.png', config)
    load_weights_of_best_model_on_validation(model, config, suffix='classifier')
    return model, (train_dataset, val_dataset, test_dataset, diffs_field)


def show_examples(model, dataset: Dataset, diffs_field, config, label, n=3):
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    data_iter = data.Iterator(dataset, batch_size=1, train=False,
                              sort=False,
                              sort_within_batch=False, repeat=False, shuffle=False,
                              device=config['DEVICE'])
    print('====START====')
    print(f'Printing examples for {label}')
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_iter, 1):
            print(f'\nOriginal src id: {batch.src_example_ids[0].data.item()}, '
                  f'edit src id: {batch.diff_example_ids[0].data.item()}')
            batch = rebatch_classifier(pad_index, batch)
            original_src = batch.original_src.cpu().numpy()[0, 1:-1]
            edit_src = batch.edit_src.cpu().numpy()[0, 1:-1]
            print("Original src: " + " ".join(lookup_words(original_src, diffs_field.vocab)))
            print("Edit src    : " + " ".join(lookup_words(edit_src, diffs_field.vocab)))
            print(f'Ground truth: {batch.trg.data.item()}')
            out = model.predict(batch)
            print(f'Predicted   : {torch.round(out).long().data.item()} ({out.data.item()})')
            if i == n:
                break
    print('====END====')


def test_classifier(model, classifier_data, config):
    print(f'\n====TESTING CLASSIFIER====\n')
    train_dataset, val_dataset, test_dataset, diffs_field = classifier_data
    print(f'Dataset sizes train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
    show_examples(model, test_dataset, diffs_field, config, label='test dataset')
    show_examples(model, val_dataset, diffs_field, config, label='val dataset')
    show_examples(model, train_dataset, diffs_field, config, label='train dataset')
    print('\n====METRICS====\n')
    classifier_metrics(model, test_dataset, diffs_field, config, label='test dataset')
    classifier_metrics(model, val_dataset, diffs_field, config, label='val dataset')
    classifier_metrics(model, train_dataset, diffs_field, config, label='train dataset')


def main(calculate_matrix=False):
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("arguments: <results_root_dir> <is_test (optional, default false)>.")
        return
    results_root_dir = sys.argv[1]
    is_test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    model, (train_dataset, _, _, diffs_field), config = load_all(results_root_dir, is_test)
    # train_dataset = take_part_from_dataset(train_dataset, 10)  # TODO: remove
    if calculate_matrix:
        matrix, diff_example_ids = get_matrix(model, train_dataset, diffs_field, config)
        write_matrix(matrix, diff_example_ids, config)
    else:
        matrix, diff_example_ids = load_matrix(results_root_dir)
    # matrix = np.random.randint(low=0, high=2, size=matrix.shape)  # TODO: remove
    print_matrix_statistics(matrix, diff_example_ids)
    model, classifier_data = \
        run_train_classifier(train_dataset, diffs_field, matrix, diff_example_ids, config)
    test_classifier(model, classifier_data, config)


if __name__ == "__main__":
    main()

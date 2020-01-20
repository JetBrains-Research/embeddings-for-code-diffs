import os
import pickle
import pprint
import sys
from pathlib import Path
from typing import Tuple, List

import torch
from torch import nn
from torchtext import data
from torchtext.data import Field, Dataset

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.SimpleLossCompute import SimpleLossCompute
from neural_editor.seq2seq.datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.datasets.dataset_utils import load_datasets
from neural_editor.seq2seq.test_utils import save_perplexity_plot
from neural_editor.seq2seq.train_utils import output_accuracy_on_data
from neural_editor.seq2seq.config import load_config, Config
from neural_editor.seq2seq.train_utils import print_data_info, make_model, \
    run_epoch, rebatch, print_examples


def load_data(verbose: bool, config: Config) -> Tuple[Dataset, Dataset, Dataset, Field]:
    diffs_field: Field = data.Field(batch_first=True, lower=config['LOWER'], include_lengths=True,
                                    unk_token=config['UNK_TOKEN'], pad_token=config['PAD_TOKEN'],
                                    init_token=config['SOS_TOKEN'],
                                    eos_token=config['EOS_TOKEN'])  # TODO: init_token=None?
    train_data, val_data, test_data = load_datasets(CodeChangesTokensDataset,
                                                    config['DATASET_ROOT'], diffs_field, config,
                                                    filter_pred=lambda x: len(vars(x)['src']) <= config[
                                                        'TOKENS_CODE_CHUNK_MAX_LEN'] and
                                                                          len(vars(x)['trg']) <= config[
                                                                              'TOKENS_CODE_CHUNK_MAX_LEN'])
    # TODO: consider building 2 vocabularies: one for (src, trg), second for diffs
    diffs_field.build_vocab(train_data.src, train_data.trg,
                            train_data.diff_alignment, train_data.diff_prev,
                            train_data.diff_updated, min_freq=config['TOKEN_MIN_FREQ'])
    if verbose:
        print_data_info(train_data, val_data, test_data, diffs_field, config)
    return train_data, val_data, test_data, diffs_field


def load_tufano_dataset(path: str, diffs_field: Field, config: Config) -> Tuple[Dataset, Dataset, Dataset]:
    train_dataset, val_dataset, test_dataset = load_datasets(CodeChangesTokensDataset,
                                                             path, diffs_field, config)
    return train_dataset, val_dataset, test_dataset


def train(model: EncoderDecoder,
          train_data: Dataset, val_data: Dataset, diffs_field: Field, config: Config) -> Tuple[List[float], List[float]]:
    """
    :param model: model to train
    :param train_data: train data
    :param val_data: validation data
    :param diffs_field: Field object from torchtext, stores vocabulary
    :param config: config of execution
    :return: train and validation perplexities for each epoch
    """
    # TODO: why it is 0, maybe padding doesn't work because no tokenizing
    # optionally add label smoothing; see the Annotated Transformer
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    train_iter = data.BucketIterator(train_data, batch_size=config['BATCH_SIZE'], train=True,
                                     shuffle=True,
                                     sort_within_batch=True,
                                     sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                     device=config['DEVICE'])
    train_batches_num: int = len(train_iter)
    train_loss_function = SimpleLossCompute(model.generator, criterion, optimizer)
    train_perplexities = []

    val_iter = data.Iterator(val_data, batch_size=config['VAL_BATCH_SIZE'], train=False,
                             sort_within_batch=True,
                             sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                             device=config['DEVICE'])
    val_iter_for_print_examples = data.Iterator(val_data, batch_size=1, train=False,
                                                sort=False, repeat=False, device=config['DEVICE'])
    val_batches_num = len(val_iter)
    # noinspection PyTypeChecker
    # reason: None is not a type of Optimizer
    val_loss_function = SimpleLossCompute(model.generator, criterion, None)
    val_perplexities = []

    epochs_num: int = config['MAX_NUM_OF_EPOCHS']
    min_val_perplexity: float = 1000000
    num_not_decreasing_steps: int = 0
    early_stopping_rounds: int = config['EARLY_STOPPING_ROUNDS']
    for epoch in range(epochs_num):
        if num_not_decreasing_steps == early_stopping_rounds:
            print(f'Training was early stopped on epoch {epoch} with early stopping rounds {early_stopping_rounds}')
            break

        print(f'Epoch {epoch} / {epochs_num}')
        model.train()
        train_perplexity = run_epoch((rebatch(pad_index, b, config) for b in train_iter),
                                     model, train_loss_function,
                                     train_batches_num,
                                     print_every=config['PRINT_EVERY_iTH_BATCH'])
        print(f'Train perplexity: {train_perplexity}')
        train_perplexities.append(train_perplexity)

        model.eval()
        with torch.no_grad():
            print_examples((rebatch(pad_index, x, config) for x in val_iter_for_print_examples),
                           model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
                           diffs_field.vocab, config, n=3)

            val_perplexity = run_epoch((rebatch(pad_index, t, config) for t in val_iter),
                                       model, val_loss_function,
                                       val_batches_num, print_every=config['PRINT_EVERY_iTH_BATCH'])
            print(f'Validation perplexity: {val_perplexity}')
            val_perplexities.append(val_perplexity)
            if val_perplexity < min_val_perplexity:
                save_model(model, 'best_on_validation', config)
                min_val_perplexity = val_perplexity
                num_not_decreasing_steps = 0
            else:
                num_not_decreasing_steps += 1

        if epoch % config['SAVE_MODEL_EVERY'] == 0:
            save_data_on_checkpoint(model, train_perplexities, val_perplexities, config)

    return train_perplexities, val_perplexities


def save_model(model: nn.Module, model_suffix: str, config: Config) -> None:
    torch.save(model.state_dict(), os.path.join(config['OUTPUT_PATH'], f'model_state_dict_{model_suffix}.pt'))
    torch.save(model, os.path.join(config['OUTPUT_PATH'], f'model_{model_suffix}.pt'))
    print(f'Model saved {model_suffix}!')


def load_weights_of_best_model_on_validation(model: nn.Module, config: Config) -> None:
    model.load_state_dict(torch.load(os.path.join(config['OUTPUT_PATH'], f'model_state_dict_best_on_validation.pt')))


def save_data_on_checkpoint(model: nn.Module,
                            train_perplexities: List[float], val_perplexities: List[float],
                            config: Config) -> None:
    save_model(model, 'checkpoint', config)
    with open(os.path.join(config['OUTPUT_PATH'], 'train_perplexities.pkl'), 'wb') as train_file:
        pickle.dump(train_perplexities, train_file)
    with open(os.path.join(config['OUTPUT_PATH'], 'val_perplexities.pkl'), 'wb') as val_file:
        pickle.dump(val_perplexities, val_file)


def test_on_unclassified_data(model: EncoderDecoder,
                              train_data: Dataset, val_data: Dataset, test_data: Dataset,
                              diffs_field: Field, config: Config) -> None:
    """
    :param train_data: train data to print some examples
    :param val_data: validation data to print some examples
    :param model: model to test
    :param test_data: test data
    :param diffs_field: Field object from torchtext
    :param config: config of execution
    :return: perplexity on test data
    """
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    test_iter = data.Iterator(test_data, batch_size=config['TEST_BATCH_SIZE'], train=False,
                              sort_within_batch=True,
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              repeat=False,
                              device=config['DEVICE'])
    test_batches_num = len(test_iter)
    # noinspection PyTypeChecker
    # reason: None is not a type of Optimizer
    test_loss_function = SimpleLossCompute(model.generator, criterion, None)
    model.eval()
    output_accuracy_on_data(model, train_data, val_data, test_data, diffs_field.vocab, pad_index, config)
    with torch.no_grad():
        test_perplexity = run_epoch((rebatch(pad_index, t, config) for t in test_iter),
                                    model, test_loss_function,
                                    test_batches_num, print_every=-1)
        print(f'Test perplexity: {test_perplexity}')


def run_train(config: Config) -> EncoderDecoder:
    pprint.pprint(config.get_config())
    config.save()

    train_dataset, val_dataset, test_dataset, diffs_field = load_data(config['VERBOSE'], config)
    model: EncoderDecoder = make_model(len(diffs_field.vocab),
                                       edit_representation_size=config['EDIT_REPRESENTATION_SIZE'],
                                       emb_size=config['WORD_EMBEDDING_SIZE'],
                                       hidden_size_encoder=config['ENCODER_HIDDEN_SIZE'],
                                       hidden_size_decoder=config['DECODER_HIDDEN_SIZE'],
                                       num_layers=config['NUM_LAYERS'],
                                       dropout=config['DROPOUT'],
                                       use_bridge=config['USE_BRIDGE'],
                                       config=config)
    # noinspection PyTypeChecker
    # reason: PyCharm doesn't understand that EncoderDecoder is child of nn.Module
    train_perplexities, val_perplexities = train(model, train_dataset, val_dataset, diffs_field, config)
    print(train_perplexities)
    print(val_perplexities)
    save_data_on_checkpoint(model, train_perplexities, val_perplexities, config)
    save_perplexity_plot([train_perplexities, val_perplexities], ['train', 'validation'], 'loss.png', config)
    load_weights_of_best_model_on_validation(model, config)
    return model


def main():
    is_test = len(sys.argv) > 1 and sys.argv[1] == 'test'
    config_path = None if len(sys.argv) < 3 else Path(sys.argv[2])
    config = load_config(is_test, config_path)
    run_train(config)


if __name__ == "__main__":
    main()

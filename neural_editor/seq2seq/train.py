import os
import pickle
import pprint
import sys
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch import nn
from torchtext import data
from torchtext.data import Field, Dataset

from datasets.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from edit_representation.sequence_encoding import EditEncoder
from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.SimpleLossCompute import SimpleLossCompute
from datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.experiments.BleuCalculation import BleuCalculation
from neural_editor.seq2seq.test_utils import save_perplexity_plot, save_metric_plot
from neural_editor.seq2seq.train_utils import output_accuracy_on_data, create_greedy_decode_method_with_batch_support, \
    calculate_top_k_accuracy
from neural_editor.seq2seq.config import load_config, Config
from neural_editor.seq2seq.train_utils import make_model, \
    run_epoch, rebatch, print_examples


def train(model: EncoderDecoder,
          train_data: Dataset, val_data: Dataset,
          fields: Tuple[Field, Field, Field],
          suffix_for_saving: str, config: Config) -> Dict[str, List[float]]:
    """
    :param model: model to train
    :param train_data: train data
    :param val_data: validation data
    :param diffs_field: Field object from torchtext, stores vocabulary
    :param config: config of execution
    :return: train and validation perplexities for each epoch
    """
    # optionally add label smoothing; see the Annotated Transformer
    src_pad_index: int = fields[0].vocab.stoi[config['PAD_TOKEN']]
    trg_pad_index: int = fields[1].vocab.stoi[config['PAD_TOKEN']]
    diff_pad_index: int = fields[2].vocab.stoi[config['PAD_TOKEN']]
    assert (src_pad_index == trg_pad_index)
    assert (trg_pad_index == diff_pad_index)
    pad_index = src_pad_index
    trg_vocab = fields[1].vocab
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['LEARNING_RATE'])

    train_iter = data.BucketIterator(train_data, batch_size=config['BATCH_SIZE'], train=True,
                                     shuffle=True,
                                     sort_within_batch=True,
                                     sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                     device=config['DEVICE'])
    train_batches_num: int = len(train_iter)
    train_loss_function = SimpleLossCompute(model.generator, criterion, optimizer)

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
    train_logs = {
        'train_ppl': [],
        'val_ppl': [],
        'val_bleu': [],
        'val_acc': []
    }
    decode_method = create_greedy_decode_method_with_batch_support(
        model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
        trg_vocab.stoi[config['SOS_TOKEN']],
        trg_vocab.stoi[config['EOS_TOKEN']],
        trg_vocab.stoi[config['UNK_TOKEN']],
        len(trg_vocab)
    )
    bleu_caclculator = BleuCalculation(config)

    epochs_num: int = config['MAX_NUM_OF_EPOCHS']
    best_metric_value: float = -1000000
    num_not_decreasing_steps: int = 0
    early_stopping_rounds: int = config['EARLY_STOPPING_ROUNDS']
    for epoch in range(epochs_num):
        if num_not_decreasing_steps == early_stopping_rounds:
            print(f'Training was early stopped on epoch {epoch} with early stopping rounds {early_stopping_rounds}')
            break

        print(f'Epoch {epoch} / {epochs_num}')
        model.train()
        train_perplexity = run_epoch((rebatch(pad_index, b, train_data, config) for b in train_iter),
                                     model, train_loss_function,
                                     train_batches_num,
                                     print_every=config['PRINT_EVERY_iTH_BATCH'])
        print(f'Train perplexity: {train_perplexity}')
        train_logs['train_ppl'].append(train_perplexity)

        model.eval()
        with torch.no_grad():
            print_examples((rebatch(pad_index, x, val_data, config) for x in val_iter_for_print_examples),
                           model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
                           fields[0].vocab, fields[1].vocab, config, n=3)

            # TODO: consider if we should or not use teacher forcing on validation
            val_perplexity = run_epoch((rebatch(pad_index, t, val_data, config) for t in val_iter),
                                       model, val_loss_function,
                                       val_batches_num, print_every=config['PRINT_EVERY_iTH_BATCH'])
            correct_all_k, total, max_top_k_predicted = calculate_top_k_accuracy([1],
                                                                                 [rebatch(pad_index, batch, val_data,
                                                                                          config) for batch in
                                                                                  val_iter],
                                                                                 decode_method, trg_vocab,
                                                                                 trg_vocab.stoi[config['EOS_TOKEN']])
            train_logs['val_ppl'].append(val_perplexity)
            train_logs['val_acc'].append(correct_all_k[0] / total)
            train_logs['val_bleu'].append(bleu_caclculator.get_bleu_score(max_top_k_predicted, val_data))
            print(f"Validation perplexity: {train_logs['val_ppl'][-1]}")
            print(f"Validation accuracy: {train_logs['val_acc'][-1]}")
            print(f"Validation BLEU: {train_logs['val_bleu'][-1]}")
            if config['BEST_ON'] == 'PPL':
                value_to_check = -train_logs['val_ppl'][-1]
            elif config['BEST_ON'] == 'BLEU':
                value_to_check = train_logs['val_bleu'][-1]
            else:
                value_to_check = train_logs['val_acc'][-1]
            if value_to_check > best_metric_value:
                save_model(model, f'best_on_validation_{suffix_for_saving}', config)
                best_metric_value = value_to_check
                num_not_decreasing_steps = 0
            else:
                num_not_decreasing_steps += 1

        if epoch % config['SAVE_MODEL_EVERY'] == 0:
            save_data_on_checkpoint(model, train_logs, suffix_for_saving, config)

    return train_logs


def save_model(model: nn.Module, model_suffix: str, config: Config) -> None:
    torch.save(model.state_dict(), os.path.join(config['OUTPUT_PATH'], f'model_state_dict_{model_suffix}.pt'))
    torch.save(model, os.path.join(config['OUTPUT_PATH'], f'model_{model_suffix}.pt'))
    print(f'Model saved {model_suffix}!')


def load_weights_of_best_model_on_validation(model: nn.Module, suffix: str, config: Config) -> None:
    model.load_state_dict(torch.load(os.path.join(config['OUTPUT_PATH'],
                                                  f'model_state_dict_best_on_validation_{suffix}.pt')))


def save_data_on_checkpoint(model: nn.Module,
                            train_logs: Dict[str, List[float]],
                            suffix: str,
                            config: Config) -> None:
    save_model(model, f'checkpoint_{suffix}', config)
    with open(os.path.join(config['OUTPUT_PATH'], f'train_logs_{suffix}.pkl'), 'wb') as train_logs_file:
        pickle.dump(train_logs, train_logs_file)


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


def run_train(train_dataset: Dataset, val_dataset: Dataset,
              fields: Tuple[Field, Field, Field],
              suffix_for_saving: str,
              edit_encoder: EditEncoder, config: Config, only_make_model=False) -> EncoderDecoder:
    pprint.pprint(config.get_config())
    config.save()

    model: EncoderDecoder = make_model(len(fields[0].vocab),
                                       len(fields[1].vocab),
                                       fields[1].vocab.unk_index,
                                       edit_encoder=edit_encoder,
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
    if only_make_model:
        # for debugging purposes only
        return model
    train_logs = train(model, train_dataset, val_dataset, fields, suffix_for_saving, config)
    print(train_logs)
    save_data_on_checkpoint(model, train_logs, suffix_for_saving, config)
    save_perplexity_plot([train_logs['train_ppl'], train_logs['val_ppl']], ['train', 'validation'],
                         f'loss_{suffix_for_saving}.png', config)
    save_metric_plot(train_logs['val_bleu'], 'bleu',
                         f'bleu_{suffix_for_saving}.png', config)
    save_metric_plot(train_logs['val_acc'], 'accuracy',
                     f'accuracy_{suffix_for_saving}.png', config)
    load_weights_of_best_model_on_validation(model, suffix_for_saving, config)
    return model


def main():
    is_test = len(sys.argv) > 1 and sys.argv[1] == 'test'
    config_path = None if len(sys.argv) < 3 else Path(sys.argv[2])
    config = load_config(is_test, config_path)
    print('\n====STARTING TRAINING OF NEURAL EDITOR====\n', end='')
    train_dataset, val_dataset, test_dataset, diffs_field = \
        CodeChangesTokensDataset.load_data(config['VERBOSE'], config)
    fields = (diffs_field, diffs_field, diffs_field)
    neural_editor = run_train(train_dataset, val_dataset, fields,
                              'neural_editor', edit_encoder=None, config=config,
                              only_make_model=not config['USE_EDIT_REPRESENTATION'])
    print('\n====STARTING TRAINING OF COMMIT MESSAGE GENERATOR====\n', end='')
    train_dataset_commit, val_dataset_commit, test_dataset_commit, fields_commit = \
        CommitMessageGenerationDataset.load_data(diffs_field, config['VERBOSE'], config)
    commit_message_generator = run_train(train_dataset_commit, val_dataset_commit, fields_commit,
                                         'commit_msg_generator', neural_editor.edit_encoder, config=config)
    return neural_editor, commit_message_generator


if __name__ == "__main__":
    main()

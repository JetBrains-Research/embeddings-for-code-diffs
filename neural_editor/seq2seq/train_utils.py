import math
import os
import pickle
import time
import typing
from datetime import timedelta
from typing import Dict, List

import numpy as np
import torch
from termcolor import colored
from torch import nn
from torchtext import data
from torchtext.data import Dataset
from torchtext.vocab import Vocab

from edit_representation.sequence_encoding.EditEncoder import EditEncoder
from neural_editor.seq2seq import SimpleLossCompute
from neural_editor.seq2seq.EncoderPredictor import EncoderPredictor
from neural_editor.seq2seq.BahdanauAttention import BahdanauAttention
from neural_editor.seq2seq.Batch import Batch, rebatch
from neural_editor.seq2seq.EncoderDecoder import EncoderDecoder
from neural_editor.seq2seq.Generator import Generator
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.Decoder import Decoder
from neural_editor.seq2seq.encoder.Encoder import Encoder
from neural_editor.seq2seq.encoder.SrcEncoder import SrcEncoder


def make_model(src_vocab_size: int, vocab_size: int, unk_index: int,
               edit_representation_size: int,
               emb_size: int,
               hidden_size_encoder: int, hidden_size_decoder: int,
               num_layers: int,
               dropout: float,
               use_bridge: bool,
               config: Config) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters."""
    attention = BahdanauAttention(hidden_size_decoder, key_size=2 * hidden_size_encoder, query_size=hidden_size_decoder)
    generator = Generator(hidden_size_decoder, vocab_size)
    embedding = nn.Embedding(src_vocab_size, emb_size)
    edit_encoder = EditEncoder(embedding, 3 * emb_size, edit_representation_size, num_layers, dropout)
    src_encoder = SrcEncoder(embedding, emb_size, hidden_size_encoder, num_layers=num_layers, dropout=dropout)
    encoder = Encoder(src_encoder, edit_encoder, config)
    model: EncoderDecoder = EncoderDecoder(
        encoder,
        Decoder(generator, embedding, emb_size, edit_representation_size,
                hidden_size_encoder, hidden_size_decoder, vocab_size, unk_index,
                attention,
                num_layers=num_layers, teacher_forcing_ratio=config['TEACHER_FORCING_RATIO'],
                dropout=dropout, bridge=use_bridge,
                use_edit_representation=config['USE_EDIT_REPRESENTATION'],
                use_copying_mechanism=config['USE_COPYING_MECHANISM']),
        embedding, generator)
    model.to(config['DEVICE'])
    return model


def make_predictor(encoder: Encoder, config: Config) -> EncoderPredictor:
    if config['FREEZE_EDIT_ENCODER_WEIGHTS']:
        freeze_weights(encoder)

    model: EncoderDecoder = EncoderPredictor(encoder)
    model.to(config['DEVICE'])
    return model


def freeze_weights(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def run_epoch(data_iter: typing.Generator, model: EncoderDecoder, loss_compute: SimpleLossCompute,
              batches_num: int, print_every: int) -> float:
    """
    Standard Training and Logging Function
    :return: loss per token
    """

    start = time.time()
    epoch_start = start
    total_tokens = 0
    total_loss = 0
    losses = []
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        out, _, pre_output, p_gen, attn_probs = model.forward(batch)
        current_losses = loss_compute((pre_output, p_gen, attn_probs), batch, batch.nseqs)
        losses += current_losses
        total_loss += np.sum(current_losses)
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if i % print_every == 0:
            elapsed = time.time() - start
            loss_mean = np.mean(current_losses)
            loss_std = np.std(current_losses)
            print(f'Epoch Step: {i} / {batches_num} '
                  f'Loss (mean): {loss_mean} '
                  f'Loss (std): {loss_std} '
                  f'Loss ratio: {loss_std / loss_mean} '
                  f'Tokens per Sec: {print_tokens / elapsed}')
            start = time.time()
            print_tokens = 0
    epoch_duration = time.time() - epoch_start
    losses_mean = np.mean(losses)
    losses_std = np.std(losses)
    print(f'Mean loss per sample: {losses_mean} Std loss per sample: {losses_std} ratio: {losses_std / losses_mean}')
    print(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
    return math.exp(total_loss / float(total_tokens))


def create_greedy_decode_method_with_batch_support(model: EncoderDecoder,
                                                   max_len: int,
                                                   sos_index: int, eos_index: int,
                                                   unk_index: int, vocab_size: int):
    def decode(batch: Batch) -> typing.List[typing.List[np.array]]:
        predicted = greedy_decode(model, batch, max_len, sos_index, eos_index, unk_index, vocab_size)
        return [[el] for el in predicted]
    return decode


def greedy_decode(model: EncoderDecoder, batch: Batch,
                  max_len: int,
                  sos_index: int, eos_index: int,
                  unk_index: int, vocab_size: int) -> typing.List[np.array]:
    """
    Greedily decode a sentence.
    :return: [DecodedSeqLenCutWithEos]
    """
    # [B, SrcSeqLen], [B, 1, SrcSeqLen], [B]
    src, src_mask, src_lengths = batch.src, batch.src_mask, batch.src_lengths
    with torch.no_grad():
        edit_final, encoder_output, encoder_final = model.encode(batch)
        prev_y = torch.ones(batch.nseqs, 1).fill_(sos_index).type_as(src)  # [B, 1]
        trg_mask = torch.ones_like(prev_y)  # [B, 1]

    output = torch.zeros((batch.nseqs, max_len))
    states = None

    for i in range(max_len):
        with torch.no_grad():
            # pre_output: [B, TrgSeqLen, DecoderH]
            out, states, pre_output, p_gen, attn_probs = model.decode(batch, edit_final, encoder_output, encoder_final,
                                                                      src_mask, prev_y, trg_mask, states)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator((pre_output, p_gen, attn_probs), batch)[:, -1]  # [B, V]

        _, next_words = torch.max(prob, dim=1)
        output[:, i] = next_words
        prev_y[:, 0] = next_words
        prev_y[prev_y >= vocab_size] = unk_index

    output = output.cpu().long().numpy()
    return remove_eos(output, eos_index)


def remove_eos(batch: np.array, eos_index: int) -> typing.List[np.array]:
    result = []
    for sequence in batch:
        eos = np.where(sequence == eos_index)[0]
        if eos.shape[0] > 0:
            sequence = sequence[:eos[0]]
        result.append(sequence)
    return result


def lookup_words(x: np.array, vocab: Vocab, oov_vocab_reverse: typing.Dict[int, str]) -> typing.List[str]:
    """
    :param x: [SeqLen]
    :param vocab: torchtext vocabulary
    :return: list of words
    """
    lookup_table = {**{i: el for i, el in enumerate(vocab.itos)}, **oov_vocab_reverse}
    return [lookup_table[i] for i in x]


def lookup_words_no_copying_mechanism(x: np.array, vocab: Vocab) -> typing.List[str]:
    """
    :param x: [SeqLen]
    :param vocab: torchtext vocabulary
    :return: list of words
    """
    lookup_table = {i: el for i, el in enumerate(vocab.itos)}
    return [lookup_table[i] for i in x]

# TODO: unite this method with print_examples
def print_examples_decode_method(example_iter: typing.Iterable, model: EncoderDecoder,
                                 vocab: Vocab, config: Config,
                                 n: int, color=None, decode_method=None) -> None:
    """Prints N examples. Assumes batch size of 1."""
    model.eval()
    count = 0

    sos_index = vocab.stoi[config['SOS_TOKEN']]
    eos_index = vocab.stoi[config['EOS_TOKEN']]

    # TODO: find out the best way to deal with <s> and </s>
    for i, batch in enumerate(example_iter):

        src = batch.scatter_indices.cpu().numpy()[0, :]
        trg = batch.trg_y_extended_vocab.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos_index else src
        trg = trg[:-1] if trg[-1] == eos_index else trg

        # remove <s> for src
        src = src[1:] if src[0] == sos_index else src

        print(colored("Example #%d" % (i + 1), color))
        # TODO_DONE: why does it have <unk>? because vocab isn't build from validation data
        print(colored("Src : " + " ".join(lookup_words(src, vocab, batch.oov_vocab_reverse))))
        print(colored("Trg : " + " ".join(lookup_words(trg, vocab, batch.oov_vocab_reverse))))

        result = decode_method(batch)
        if len(result[0]) == 0:
            # this if can be true if none of the sequences didn't terminate
            print(colored("Pred: "))
        else:
            result = result[0][0]
            print(colored("Pred: " + " ".join(lookup_words(result, vocab, batch.oov_vocab_reverse))))

        count += 1
        if count == n:
            break


def print_examples(example_iter: typing.Iterable, model: EncoderDecoder,
                   max_len: int, src_vocab: Vocab, trg_vocab: Vocab, config: Config,
                   n: int, color=None) -> None:
    """Prints N examples. Assumes batch size of 1."""
    model.eval()
    count = 0

    assert (src_vocab.stoi[config['SOS_TOKEN']] == trg_vocab.stoi[config['SOS_TOKEN']])
    assert (src_vocab.stoi[config['EOS_TOKEN']] == trg_vocab.stoi[config['EOS_TOKEN']])
    sos_index = src_vocab.stoi[config['SOS_TOKEN']]
    eos_index = src_vocab.stoi[config['EOS_TOKEN']]

    # TODO: find out the best way to deal with <s> and </s>
    for i, batch in enumerate(example_iter):

        src = batch.scatter_indices.cpu().numpy()[0, :]
        trg = batch.trg_y_extended_vocab.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos_index else src
        trg = trg[:-1] if trg[-1] == eos_index else trg

        # remove <s> for src
        src = src[1:] if src[0] == sos_index else src

        result = greedy_decode(model, batch, max_len, sos_index, eos_index, trg_vocab.unk_index, len(trg_vocab))
        result = result[0]
        print(colored("Example #%d" % (i + 1), color))
        # TODO_DONE: why does it have <unk>? because vocab isn't build from validation data
        print(colored("Src : " + " ".join(lookup_words(src, trg_vocab, batch.oov_vocab_reverse))))
        print(colored("Trg : " + " ".join(lookup_words(trg, trg_vocab, batch.oov_vocab_reverse))))
        print(colored("Pred: " + " ".join(lookup_words(result, trg_vocab, batch.oov_vocab_reverse))))

        count += 1
        if count == n:
            break


def calculate_accuracy(dataset_iterator: typing.Iterable,
                       model: EncoderDecoder,
                       max_len: int,
                       vocab: Vocab,
                       config: Config) -> float:
    sos_index = vocab.stoi[config['SOS_TOKEN']]
    eos_index = vocab.stoi[config['EOS_TOKEN']]

    correct = 0
    total = 0
    for batch in dataset_iterator:
        targets = remove_eos(batch.trg_y_extended_vocab.cpu().numpy(), eos_index)

        results = greedy_decode(model, batch, max_len, sos_index, eos_index, vocab.unk_index, len(vocab))
        for i in range(len(targets)):
            if np.all(targets[i] == results[i]):
                correct += 1
            total += 1
    return correct / total


def calculate_top_k_accuracy(topk_values: typing.List[int], dataset_iterator: typing.Iterator,
                             decode_method, trg_vocab: Vocab, eos_index: int, dataset_len: int) \
        -> typing.Tuple[typing.List[int], int, typing.List[typing.List[typing.List[str]]]]:
    # TODO: handle <unk> token issue
    correct = [0 for _ in range(len(topk_values))]
    max_k = topk_values[-1]
    total = 0
    max_top_k_results: typing.List[typing.Optional[typing.List[typing.List[str]]]] = [None for _ in range(dataset_len)]
    for batch in dataset_iterator:
        targets = remove_eos(batch.trg_y_extended_vocab.cpu().numpy(), eos_index)
        results = decode_method(batch)
        for example_idx_in_batch in range(len(results)):
            example_id = batch.ids[example_idx_in_batch].item()
            target = targets[example_idx_in_batch]
            example_top_k_results = results[example_idx_in_batch][:max_k]
            decoded_tokens = [lookup_words(result, trg_vocab, batch.oov_vocab_reverse)
                              for result in example_top_k_results]
            max_top_k_results[example_id] = decoded_tokens
            tail_id = 0
            for i, result in enumerate(example_top_k_results):
                if i + 1 > topk_values[tail_id]:
                    tail_id += 1
                if len(result) == len(target) and np.all(result == target):
                    for j in range(tail_id, len(correct)):
                        correct[j] += 1
                    break
        total += len(batch)
    return correct, total, max_top_k_results


def output_accuracy_on_data(model: EncoderDecoder,
                            train_data: Dataset, val_data: Dataset, test_data: Dataset,
                            vocab: Vocab, pad_index: int, config: Config) -> None:
    with torch.no_grad():
        for dataset, label in zip([test_data, val_data, train_data], ['TEST', 'VALIDATION', 'TRAIN']):
            print_examples_iterator = data.Iterator(dataset, batch_size=1, train=False, sort=False,
                                                    repeat=False, device=config['DEVICE'])
            print(f'==={label} EXAMPLES===')
            print_examples((rebatch(x, dataset, config) for x in print_examples_iterator),
                           model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
                           vocab, vocab, config, n=3)
            accuracy_iterator = data.Iterator(dataset, batch_size=config['TEST_BATCH_SIZE'], train=False,
                                              sort_within_batch=True,
                                              sort_key=lambda x: (len(x.src), len(x.trg)),
                                              repeat=False,
                                              device=config['DEVICE'])
            accuracy = calculate_accuracy((rebatch(t, dataset, config) for t in accuracy_iterator),
                                          model,
                                          config['TOKENS_CODE_CHUNK_MAX_LEN'],
                                          vocab, config)
            print(f'Accuracy on {label}: {accuracy}')


def save_data_on_checkpoint(model: nn.Module,
                            train_logs: Dict[str, List[float]],
                            suffix: str,
                            config: Config) -> None:
    save_model(model, f'checkpoint_{suffix}', config)
    with open(os.path.join(config['OUTPUT_PATH'], f'train_logs_{suffix}.pkl'), 'wb') as train_logs_file:
        pickle.dump(train_logs, train_logs_file)


def load_weights_of_best_model_on_validation(model: nn.Module, suffix: str, config: Config) -> None:
    model.load_state_dict(torch.load(os.path.join(config['OUTPUT_PATH'],
                                                  f'model_state_dict_best_on_validation_{suffix}.pt')))


def save_model(model: nn.Module, model_suffix: str, config: Config) -> None:
    torch.save(model.state_dict(), os.path.join(config['OUTPUT_PATH'], f'model_state_dict_{model_suffix}.pt'))
    torch.save(model, os.path.join(config['OUTPUT_PATH'], f'model_{model_suffix}.pt'))
    print(f'Model saved {model_suffix}!')
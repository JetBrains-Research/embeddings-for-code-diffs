import math
import time
import typing
from datetime import timedelta
from termcolor import colored

import numpy as np
import torch
import torchtext
from torch import nn
from torchtext import data
from torchtext.data import Dataset, Field
from torchtext.vocab import Vocab

from edit_representation.sequence_encoding.EditEncoder import EditEncoder
from neural_editor.seq2seq import SimpleLossCompute
from neural_editor.seq2seq.BahdanauAttention import BahdanauAttention
from neural_editor.seq2seq.Batch import Batch
from neural_editor.seq2seq.EncoderDecoder import EncoderDecoder
from neural_editor.seq2seq.Generator import Generator
from neural_editor.seq2seq.decoder.Decoder import Decoder
from neural_editor.seq2seq.encoder.Encoder import Encoder
from neural_editor.seq2seq.config import Config


def make_model(vocab_size: int, edit_representation_size: int, emb_size: int,
               hidden_size_encoder: int, hidden_size_decoder: int,
               num_layers: int,
               dropout: float,
               use_bridge: bool,
               config: Config) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters."""
    # TODO_DONE: change hidden size of decoder
    attention = BahdanauAttention(hidden_size_decoder, key_size=2 * hidden_size_encoder, query_size=hidden_size_decoder)

    model: EncoderDecoder = EncoderDecoder(
        Encoder(emb_size, hidden_size_encoder, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, edit_representation_size,
                hidden_size_encoder, hidden_size_decoder,
                attention,
                num_layers=num_layers, dropout=dropout, bridge=use_bridge),
        EditEncoder(3 * emb_size, edit_representation_size, num_layers, dropout),
        nn.Embedding(vocab_size, emb_size),  # 1 -> Emb
        Generator(hidden_size_decoder, vocab_size))
    model.to(config['DEVICE'])
    return model


def rebatch(pad_idx: int, batch: torchtext.data.Batch, config: Config) -> Batch:
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # These fields are added dynamically by PyTorch
    return Batch(batch.src, batch.trg, batch.diff_alignment,
                 batch.diff_prev, batch.diff_updated, pad_idx, config)


def print_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset, field: Field, config: Config) -> None:
    """ This prints some useful stuff about our data sets. """

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))
    print('test', len(test_data), "\n")

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['src']))
    print("trg:", " ".join(vars(train_data[0])['trg']))
    print("diff_alignment:", " ".join(vars(train_data[0])['diff_alignment']))
    print("diff_prev:", " ".join(vars(train_data[0])['diff_prev']))
    print("diff_updated:", " ".join(vars(train_data[0])['diff_updated']), '\n')

    print("Most common words:")
    print("\n".join(["%10s %10d" % x for x in field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words:")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(field.vocab.itos[:10])), "\n")

    print("Special words frequency and ids: ")
    special_tokens = [config['UNK_TOKEN'], config['PAD_TOKEN'], config['SOS_TOKEN'], config['EOS_TOKEN'],
                      config['REPLACEMENT_TOKEN'], config['DELETION_TOKEN'], config['ADDITION_TOKEN'],
                      config['UNCHANGED_TOKEN'], config['PADDING_TOKEN']]
    for special_token in special_tokens:
        print(f"{special_token} {field.vocab.freqs[special_token]} {field.vocab.stoi[special_token]}")

    print("Number of words (types):", len(field.vocab))


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
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        out, _, pre_output = model.forward(batch)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print(f'Epoch Step: {i} / {batches_num} '
                  f'Loss: {loss / batch.nseqs} '
                  f'Tokens per Sec: {print_tokens / elapsed}')
            start = time.time()
            print_tokens = 0
    epoch_duration = time.time() - epoch_start
    print(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
    return math.exp(total_loss / float(total_tokens))


def greedy_decode(model: EncoderDecoder, batch: Batch,
                  max_len: int,
                  sos_index: int, eos_index: int) -> typing.List[np.array]:
    """
    Greedily decode a sentence.
    :return: [DecodedSeqLenCutWithEos]
    """
    # TODO: create beam search
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
            out, states, pre_output = model.decode(edit_final, encoder_output, encoder_final,
                                                   src_mask, prev_y, trg_mask, states)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])  # [B, V]

        _, next_words = torch.max(prob, dim=1)
        output[:, i] = next_words
        prev_y[:, 0] = next_words

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


def lookup_words(x: np.array, vocab: Vocab) -> typing.List[str]:
    """
    :param x: [SeqLen]
    :param vocab: torchtext vocabulary
    :return: list of words
    """
    return [vocab.itos[i] for i in x]


def print_examples(example_iter: typing.Iterable, model: EncoderDecoder,
                   max_len: int, vocab: Vocab, config: Config,
                   n: int, color=None) -> None:
    """Prints N examples. Assumes batch size of 1."""
    model.eval()
    count = 0

    sos_index = vocab.stoi[config['SOS_TOKEN']]
    eos_index = vocab.stoi[config['EOS_TOKEN']]

    # TODO: find out the best way to deal with <s> and </s>
    for i, batch in enumerate(example_iter):

        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos_index else src
        trg = trg[:-1] if trg[-1] == eos_index else trg

        # remove <s> for src
        src = src[1:] if src[0] == sos_index else src

        result = greedy_decode(model, batch, max_len, sos_index, eos_index)
        result = result[0]
        print(colored("Example #%d" % (i + 1), color))
        # TODO_DONE: why does it have <unk>? because vocab isn't build from validation data
        print(colored("Src : " + " ".join(lookup_words(src, vocab))))
        print(colored("Trg : " + " ".join(lookup_words(trg, vocab))))
        print(colored("Pred: " + " ".join(lookup_words(result, vocab))))

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
        targets = remove_eos(batch.trg_y.cpu().numpy(), eos_index)

        results = greedy_decode(model, batch, max_len, sos_index, eos_index)
        for i in range(len(targets)):
            if np.all(targets[i] == results[i]):
                correct += 1
            total += 1
    return correct / total


def output_accuracy_on_data(model: EncoderDecoder,
                            train_data: Dataset, val_data: Dataset, test_data: Dataset,
                            vocab: Vocab, pad_index: int, config: Config) -> None:
    with torch.no_grad():
        for dataset, label in zip([test_data, val_data, train_data], ['TEST', 'VALIDATION', 'TRAIN']):
            print_examples_iterator = data.Iterator(dataset, batch_size=1, train=False, sort=False,
                                                    repeat=False, device=config['DEVICE'])
            print(f'==={label} EXAMPLES===')
            print_examples((rebatch(pad_index, x, config) for x in print_examples_iterator),
                           model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
                           vocab, config, n=3)
            accuracy_iterator = data.Iterator(dataset, batch_size=config['TEST_BATCH_SIZE'], train=False,
                                              sort_within_batch=True,
                                              sort_key=lambda x: (len(x.src), len(x.trg)),
                                              repeat=False,
                                              device=config['DEVICE'])
            accuracy = calculate_accuracy((rebatch(pad_index, t, config) for t in accuracy_iterator),
                                          model,
                                          config['TOKENS_CODE_CHUNK_MAX_LEN'],
                                          vocab, config)
            print(f'Accuracy on {label}: {accuracy}')


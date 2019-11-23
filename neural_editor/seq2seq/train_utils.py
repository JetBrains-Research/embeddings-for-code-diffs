import math
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchtext
from torch import nn
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
from neural_editor.seq2seq.train_config import CONFIG


def make_model(vocab_size: int, edit_representation_size: int, emb_size: int,
               hidden_size_encoder: int, hidden_size_decoder: int,
               num_layers: int,
               dropout: float,
               use_bridge: bool) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters."""
    # TODO: change hidden size of decoder
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
    model.to(CONFIG['DEVICE'])
    return model


def rebatch(pad_idx: int, batch: torchtext.data.Batch) -> Batch:
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # These fields are added dynamically by PyTorch
    return Batch(batch.src, batch.trg, batch.diff_alignment,
                 batch.diff_prev, batch.diff_updated, pad_idx)


def print_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset, field: Field) -> None:
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
    special_tokens = [CONFIG['UNK_TOKEN'], CONFIG['PAD_TOKEN'], CONFIG['SOS_TOKEN'], CONFIG['EOS_TOKEN'],
                      CONFIG['REPLACEMENT_TOKEN'], CONFIG['DELETION_TOKEN'], CONFIG['ADDITION_TOKEN'],
                      CONFIG['UNCHANGED_TOKEN'], CONFIG['PADDING_TOKEN']]
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

    return math.exp(total_loss / float(total_tokens))


def greedy_decode(model: EncoderDecoder, batch: Batch,
                  max_len: int,
                  sos_index: int, eos_index: int) -> typing.Tuple[np.array, np.array]:
    """
    Greedily decode a sentence.
    :return: Tuple[[DecodedSeqLenCutWithEos], [1, DecodedSeqLen, SrcSeqLen]]
    """
    # [B, SrcSeqLen], [B, 1, SrcSeqLen], [B]
    src, src_mask, src_lengths = batch.src, batch.src_mask, batch.src_lengths
    with torch.no_grad():
        edit_output, edit_final, encoder_output, encoder_final = model.encode(batch)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)  # [1, 1]
        trg_mask = torch.ones_like(prev_y)  # [1, 1]

    output = []
    attention_scores = []

    for i in range(max_len):
        with torch.no_grad():
            # pre_output: [B, TrgSeqLen, DecoderH]
            out, hidden, pre_output = model.decode(edit_final, encoder_output, encoder_final,
                                                   src_mask, prev_y, trg_mask)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])  # [B, V]

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x: np.array, vocab: Vocab) -> typing.List[str]:
    """
    :param x: [SeqLen]
    :param vocab: torchtext vocabulary
    :return: list of words
    """
    return [vocab.itos[i] for i in x]


def print_examples(example_iter: typing.Generator, model: EncoderDecoder, max_len: int, vocab: Vocab, n: int) -> None:
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()

    sos_index = vocab.stoi[CONFIG['SOS_TOKEN']]
    eos_index = vocab.stoi[CONFIG['EOS_TOKEN']]

    # TODO: find out the best way to deal with <s> and </s>
    for i, batch in enumerate(example_iter):

        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos_index else src
        trg = trg[:-1] if trg[-1] == eos_index else trg

        # remove <s> for src
        src = src[1:] if src[0] == sos_index else src

        result, _ = greedy_decode(model, batch, max_len, sos_index, eos_index)
        print("Example #%d" % (i + 1))
        print("Src : ", " ".join(lookup_words(src, vocab)))  # TODO: why does it have <unk>?
        print("Trg : ", " ".join(lookup_words(trg, vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab)))
        print()

        count += 1
        if count == n:
            break


def plot_perplexity(perplexities: typing.List[float], labels: typing.List[str]) -> None:
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    for perplexity_values, label in zip(perplexities, labels):
        plt.plot(perplexity_values, label=label)
    plt.show()

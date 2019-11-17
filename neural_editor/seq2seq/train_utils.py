import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from edit_representation.sequence_encoding.EditEncoder import EditEncoder
from neural_editor.seq2seq.BahdanauAttention import BahdanauAttention
from neural_editor.seq2seq.Batch import Batch
from neural_editor.seq2seq.EncoderDecoder import EncoderDecoder
from neural_editor.seq2seq.Generator import Generator
from neural_editor.seq2seq.decoder.Decoder import Decoder
from neural_editor.seq2seq.encoder.Encoder import Encoder
from neural_editor.seq2seq.train_config import CONFIG


def make_model(vocab_size, edit_representation_size=512, emb_size=128, hidden_size_encoder=128, hidden_size_decoder=128, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    # TODO: change hidden size of decoder
    attention = BahdanauAttention(hidden_size_decoder, key_size=2 * hidden_size_encoder, query_size=hidden_size_decoder)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size_encoder, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, edit_representation_size, hidden_size_encoder, hidden_size_decoder, attention, num_layers=num_layers, dropout=dropout),
        EditEncoder(3 * emb_size, edit_representation_size, num_layers=num_layers),
        nn.Embedding(vocab_size, emb_size),
        Generator(hidden_size_decoder, vocab_size))

    return model.cuda() if CONFIG['USE_CUDA'] else model


def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg, batch.diff_alignment,
                 batch.diff_prev, batch.diff_updated, pad_idx)


def print_data_info(train_data, valid_data, test_data, field):
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


def run_epoch(data_iter, model, loss_compute, batches_num, print_every=50):
    """Standard Training and Logging Function"""

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
            print("Epoch Step: %d / %d Loss: %f Tokens per Sec: %f" %
                  (i, batches_num, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


def greedy_decode(model, batch, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    src, src_mask, src_lengths = batch.src, batch.src_mask, batch.src_lengths
    with torch.no_grad():
        (encoder_hidden, encoder_final, encoder_cell_state), _, \
        edit_representation_final, edit_representation_cell_state = model.encode(batch)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
                edit_representation_final, edit_representation_cell_state,
                encoder_hidden, encoder_final, encoder_cell_state, src_mask,
                prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

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


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


def print_examples(example_iter, model, n=2, max_len=100, src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()

    if src_vocab is not None and trg_vocab is not None:
        src_sos_index = src_vocab.stoi[CONFIG['SOS_TOKEN']]
        src_eos_index = src_vocab.stoi[CONFIG['EOS_TOKEN']]
        trg_sos_index = trg_vocab.stoi[CONFIG['SOS_TOKEN']]
        trg_eos_index = trg_vocab.stoi[CONFIG['EOS_TOKEN']]
    else:
        src_sos_index = None
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None

    # TODO: find out the best way to deal with <s> and </s>
    for i, batch in enumerate(example_iter):

        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg

        # remove <s> for src
        src = src[1:] if src[0] == src_sos_index else src

        result, _ = greedy_decode(
            model, batch,
            max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i + 1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))  # TODO: why does it have <unk>?
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()

        count += 1
        if count == n:
            break


def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    plt.show()

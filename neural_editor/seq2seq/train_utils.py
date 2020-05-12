import math
import random
import time
import typing
from collections import OrderedDict
from datetime import timedelta

import numpy as np
import torch
from termcolor import colored
from torch import nn
from torch.nn import Parameter
from torchtext import data
from torchtext.data import Dataset, Field
from torchtext.vocab import Vocab

from edit_representation.sequence_encoding.EditEncoder import EditEncoder
from neural_editor.seq2seq import SimpleLossCompute
from neural_editor.seq2seq.BahdanauAttention import BahdanauAttention
from neural_editor.seq2seq.Batch import Batch, rebatch
from neural_editor.seq2seq.EncoderDecoder import EncoderDecoder
from neural_editor.seq2seq.Generator import Generator
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.Decoder import Decoder
from neural_editor.seq2seq.encoder.Encoder import Encoder


def make_model(vocab_size: int, edit_representation_size: int, emb_size: int,
               hidden_size_encoder: int, hidden_size_decoder: int,
               num_layers: int,
               dropout: float,
               use_bridge: bool,
               config: Config) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters."""
    # TODO_DONE: change hidden size of decoder
    attention = BahdanauAttention(hidden_size_decoder, key_size=2 * hidden_size_encoder, query_size=hidden_size_decoder)

    generator = Generator(hidden_size_decoder, vocab_size)
    embedding = nn.Embedding(vocab_size, emb_size)
    model: EncoderDecoder = EncoderDecoder(
        Encoder(emb_size, hidden_size_encoder, num_layers=num_layers, dropout=dropout),
        Decoder(generator, embedding, emb_size, edit_representation_size,
                hidden_size_encoder, hidden_size_decoder,
                attention,
                num_layers=num_layers, teacher_forcing_ratio=config['TEACHER_FORCING_RATIO'],
                dropout=dropout, bridge=use_bridge,
                use_edit_representation=config['USE_EDIT_REPRESENTATION']),
        EditEncoder(3 * emb_size, edit_representation_size, num_layers, dropout),
        embedding, generator, config)
    model.to(config['DEVICE'])
    return model


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


def load_state_dict_from_jiang_model(model, state_dict: OrderedDict) -> None:
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state or name == 'decoder.embedding.weight':
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def run_epoch(data_iter: typing.List, model: EncoderDecoder, loss_compute: SimpleLossCompute,
              epoch: int, batches_num: int, print_every: int, config: Config) -> float:
    """
    Standard Training and Logging Function
    :return: loss per token
    """
    epoch_start = time.time()
    total_loss, total_tokens = 0, 0
    if config['LOSS_FUNCTION_PARAMS']['measure'] == 'epochs':
        if config['LOSS_FUNCTION_PARAMS']['default_loss_period'] != 0 and \
                epoch % config['LOSS_FUNCTION_PARAMS']['default_loss_period'] == 0:
            total_default_loss, total_tokens_default_loss = iterate_over_all_data(batches_num, data_iter,
                                                                                  loss_compute, model,
                                                                                  print_every,
                                                                                  ratios=(1, 0),
                                                                                  config=config)
            total_loss += total_default_loss
            total_tokens += total_tokens_default_loss
        if config['LOSS_FUNCTION_PARAMS']['bug_fixing_loss_period'] != 0 and \
                epoch % config['LOSS_FUNCTION_PARAMS']['bug_fixing_loss_period'] == 0:
            total_bug_fixing_loss, total_tokens_bug_fixing_loss = iterate_over_all_data(batches_num, data_iter,
                                                                                        loss_compute, model,
                                                                                        print_every,
                                                                                        ratios=(0, 1),
                                                                                        config=config)
            total_loss += total_bug_fixing_loss
            total_tokens += total_tokens_bug_fixing_loss
    elif config['LOSS_FUNCTION_PARAMS']['measure'] == 'batches':
        loss, tokens = iterate_over_all_data(batches_num, data_iter,
                                             loss_compute, model,
                                             print_every,
                                             ratios=(config['LOSS_FUNCTION_PARAMS']['default_loss_period'],
                                                     config['LOSS_FUNCTION_PARAMS']['bug_fixing_loss_period']),
                                             config=config)
        total_loss += loss
        total_tokens += tokens
    else:
        raise Exception('Unsupported measure fro LOSS_FUNCTION_PARAMS')
    epoch_duration = time.time() - epoch_start
    print(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
    return math.exp(total_loss / float(total_tokens))


def iterate_over_all_data(batches_num, data_iter, loss_compute, model, print_every, ratios, config: Config):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0
    for i, batch in enumerate(data_iter, 1):
        if config['UPDATE_TRAIN_VECTORS_EVERY_iTH_EPOCH']['measure'] == 'batches' and \
                config['UPDATE_TRAIN_VECTORS_EVERY_iTH_EPOCH']['period'] != 0 and \
                (i - 1) % config['UPDATE_TRAIN_VECTORS_EVERY_iTH_EPOCH']['period'] == 0:
            model.update_training_vectors()
        do_default_loss = random.random() < ratios[0]
        do_bug_fixing_loss = random.random() < ratios[1]
        pre_output_default_loss = None
        pre_output_bug_fixing_loss = None
        if do_default_loss:
            _, _, pre_output_default_loss = model.forward(batch, ignore_encoded_train=True)
            total_tokens += batch.ntokens
            print_tokens += batch.ntokens
        if do_bug_fixing_loss:
            _, _, pre_output_bug_fixing_loss = model.forward(batch, ignore_encoded_train=False)
            total_tokens += batch.ntokens
            print_tokens += batch.ntokens
        if pre_output_default_loss is None and pre_output_bug_fixing_loss is None:
            loss = 0
        elif pre_output_default_loss is None:
            loss = loss_compute(pre_output_bug_fixing_loss, batch.trg_y, batch.nseqs)
        elif pre_output_bug_fixing_loss is None:
            loss = loss_compute(pre_output_default_loss, batch.trg_y, batch.nseqs)
        else:
            loss = loss_compute(pre_output_default_loss, batch.trg_y, batch.nseqs, pre_output_bug_fixing_loss)
        total_loss += loss

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print(f'Epoch Step: {i} / {batches_num} '
                  f'Loss: {loss / batch.nseqs} '
                  f'Tokens per Sec: {print_tokens / elapsed}')
            start = time.time()
            print_tokens = 0
    return total_loss, total_tokens


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


def greedy_decode_top_k_edits(model: EncoderDecoder, batch: Batch,
                              max_len: int,
                              sos_index: int, eos_index: int, n_neighbors: int) -> typing.List[typing.List[np.array]]:
    """
    Greedily decode a sentence.
    :return: [DecodedSeqLenCutWithEos]
    """
    # TODO: create beam search
    # [B, SrcSeqLen], [B, 1, SrcSeqLen], [B]
    src, src_mask, src_lengths = batch.src, batch.src_mask, batch.src_lengths
    with torch.no_grad():
        (edit_finals, indices), encoder_output, encoder_final = model.encode(batch, ignore_encoded_train=False,
                                                                             n_neighbors=n_neighbors)
    result = np.empty((len(batch), n_neighbors), dtype=object)
    for edit_k, edit_final in enumerate(edit_finals):
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
        result[:, edit_k] = remove_eos(output, eos_index)
    return result, indices


def create_greedy_decode_method_top_k_edits(model: EncoderDecoder,
                                            max_len: int,
                                            sos_index: int, eos_index: int, n_neighbors: int):
    def decode(batch: Batch) -> typing.List[typing.List[np.array]]:
        predicted, _ = greedy_decode_top_k_edits(model, batch, max_len, sos_index, eos_index, n_neighbors)
        return predicted

    return decode


def create_greedy_decode_method_top_k_edits_with_indices(model: EncoderDecoder,
                                                         max_len: int,
                                                         sos_index: int, eos_index: int, n_neighbors: int):
    def decode(batch: Batch) -> typing.Tuple[typing.List[typing.List[np.array]], np.ndarray]:
        predicted, indices = greedy_decode_top_k_edits(model, batch, max_len, sos_index, eos_index, n_neighbors)
        return predicted, indices

    return decode


def create_greedy_decode_method(model: EncoderDecoder,
                                max_len: int,
                                sos_index: int, eos_index: int):
    def decode(batch: Batch) -> typing.List[typing.List[np.array]]:
        predicted = greedy_decode(model, batch, max_len, sos_index, eos_index)
        return [[el] for el in predicted]

    return decode


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

        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == eos_index else src
        trg = trg[:-1] if trg[-1] == eos_index else trg

        # remove <s> for src
        src = src[1:] if src[0] == sos_index else src

        print(colored("Example #%d" % (i + 1), color))
        # TODO_DONE: why does it have <unk>? because vocab isn't build from validation data
        print(colored("Src : " + " ".join(lookup_words(src, vocab))))
        print(colored("Trg : " + " ".join(lookup_words(trg, vocab))))

        result = decode_method(batch)
        if len(result[0]) == 0:
            # this if can be true if none of the sequences didn't terminate
            print(colored("Pred: "))
        else:
            result = result[0][0]
            print(colored("Pred: " + " ".join(lookup_words(result, vocab))))

        count += 1
        if count == n:
            break


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


def get_greedy_correct_predicted_examples(dataset_iterator: typing.List,
                                          model: EncoderDecoder,
                                          max_len: int,
                                          sos_index: int,
                                          eos_index: int,
                                          print_every=2) -> np.ndarray:
    correct_ids = []
    start = time.time()
    for i_batch, batch in enumerate(dataset_iterator):
        targets = remove_eos(batch.trg_y.cpu().numpy(), eos_index)

        results = greedy_decode(model, batch, max_len, sos_index, eos_index)
        for i in range(len(targets)):
            if len(targets[i]) == len(results[i]) and np.all(targets[i] == results[i]):
                correct_ids.append(batch.ids[i].item())
        if (i_batch + 1) % print_every == 0:
            end = time.time()
            duration = end - start
            start = end
            print(f'Processing {i_batch + 1} / {len(dataset_iterator)}, '
                  f'{int((i_batch + 1) / print_every)} / {math.ceil(len(dataset_iterator) / print_every)}, '
                  f'elapsed: {str(timedelta(seconds=duration))}')
    return np.array(correct_ids)


def calculate_top_k_accuracy(topk_values: typing.List[int], dataset_iterator: typing.Iterator,
                             decode_method, trg_vocab: Vocab, eos_index: int, dataset_len: int) \
        -> typing.Tuple[typing.List[int], int, typing.List[typing.List[typing.List[str]]]]:
    correct = [0] * len(topk_values)
    max_k = topk_values[-1]
    total = 0
    max_top_k_results: typing.List[typing.Optional[typing.List[typing.List[str]]]] = [None] * dataset_len
    for batch in dataset_iterator:
        targets = remove_eos(batch.trg_y.cpu().numpy(), eos_index)
        results = decode_method(batch)
        for example_idx_in_batch in range(len(results)):
            example_id = batch.ids[example_idx_in_batch].item()
            target = targets[example_idx_in_batch]
            example_top_k_results = results[example_idx_in_batch][:max_k]
            decoded_tokens = [lookup_words(result, trg_vocab)
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
            print_examples((rebatch(pad_index, x, config) for x in print_examples_iterator),
                           model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
                           vocab, config, n=3)
            accuracy_iterator = data.Iterator(dataset, batch_size=config['TEST_BATCH_SIZE'], train=False,
                                              sort_within_batch=True,
                                              sort_key=lambda x: (len(x.src), len(x.trg)),
                                              repeat=False,
                                              device=config['DEVICE'])
            accuracy = get_greedy_correct_predicted_examples((rebatch(pad_index, t, config) for t in accuracy_iterator),
                                                             model,
                                                             config['TOKENS_CODE_CHUNK_MAX_LEN'],
                                                             vocab, config)
            print(f'Accuracy on {label}: {accuracy}')

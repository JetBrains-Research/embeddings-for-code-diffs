from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from neural_editor.seq2seq import EncoderDecoder, Batch
from neural_editor.seq2seq.config import Config


class BatchedBeamSearch:
    def __init__(self, beam_size: int, model: EncoderDecoder,
                 sos_index: int, eos_index: int,
                 config: Config) -> None:
        """
        Constructor of beam search that supports decoding by batches.
        :param beam_size: beam search width
        :param model: seq2seq encoder decoder model
        :param sos_index: start of sequence symbol
        :param eos_index: end of sequence symbol
        :param config: config of execution
        """
        super().__init__()
        self.beam_size = beam_size
        self.model = model
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_len = config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1
        self.beam_indexing = torch.arange(self.beam_size)
        self.config = config

    def first_step(self, batch: Batch):
        """
        Method that makes first decoding step and returns all variables needed to continue decoding.
        It is useful because we need to take first most probable symbols
        and go to tensor of probabilities of shape [B, beam_size].
        :param batch: batch to process.
        :return: tensors and states needed to decode further.
        """
        src, src_mask, src_lengths = batch.src, batch.src_mask, batch.src_lengths
        edit_final, encoder_output, encoder_final = self.model.encode(batch)
        prev_y = torch.ones(len(batch), 1).fill_(self.sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

        out, states, pre_output = self.model.decode(edit_final, encoder_output, encoder_final,
                                                    src_mask, prev_y, trg_mask, states_to_initialize=None)
        prob = self.model.generator(pre_output[:, -1])

        return edit_final, encoder_output, encoder_final, states, prob, prev_y, trg_mask, src_mask

    def decode(self, batch: Batch) -> List[np.array]:
        """
        Method decodes a batch and returns list of answers for each example in batch.
        :param batch: batch to decode.
        :return: list of numpy arrays, each numpy array contains sequences in descending by probability order,
        length of guesses may vary and guesses doesn't contain sos and eos and pad tokens
        """
        batch_size = len(batch)
        edit_final, encoder_output, encoder_final, states, prob, prev_y, trg_mask, src_mask = self.first_step(batch)
        vocab_size = prob.shape[-1]

        edit_final, encoder_final, states = self.reshape_states([edit_final, encoder_final, states])
        encoder_output, prev_y, trg_mask, src_mask = self.reshape_batches([encoder_output, prev_y, trg_mask, src_mask])
        current_sequences = torch.zeros((batch_size, self.beam_size, self.max_len)).long().to(self.config['DEVICE'])
        best_probs, next_words = torch.sort(prob, dim=1, descending=True)
        best_probs, next_words = best_probs[:, :self.beam_size], next_words[:, :self.beam_size]
        current_sequences[:, :, 0] = next_words
        prev_y[:, 0] = next_words.flatten()

        results = [[] for _ in range(batch_size)]
        results_probs = [[] for _ in range(batch_size)]
        new_best_permutation = torch.zeros(batch_size, self.beam_size).long().to(self.config['DEVICE'])
        new_next_words = torch.zeros(batch_size, self.beam_size).long().to(self.config['DEVICE'])
        new_best_probs = torch.zeros(batch_size, self.beam_size).to(self.config['DEVICE'])

        for i in range(1, self.max_len):
            out, states, pre_output = self.model.decode(edit_final, encoder_output, encoder_final,
                                                        src_mask, prev_y, trg_mask, states)
            prob = self.model.generator(pre_output[:, -1])  # [B * beam_size, V]
            prob = unstick_two_dimensions_together(prob, dim_value=batch_size, dim=0).reshape(batch_size,
                                                                                              self.beam_size,
                                                                                              vocab_size)
            new_best = best_probs.unsqueeze(2).repeat((1, 1, prob.shape[2])) + prob
            new_best = stick_two_dimensions_together(new_best, dim=1)
            best_probs, sorted_indices = torch.sort(new_best, dim=1, descending=True)

            all_permutations = sorted_indices // vocab_size
            next_words = sorted_indices % vocab_size
            # TODO: optimize this loop
            for batch_id in range(batch_size):
                collected = 0
                ind = 0
                while collected != self.beam_size:
                    if next_words[batch_id][ind] == self.eos_index:
                        results[batch_id].append(
                            current_sequences[batch_id][all_permutations[batch_id][ind]][:i].detach().long().numpy())
                        results_probs[batch_id].append(best_probs[batch_id][ind].detach())
                    else:
                        new_best_permutation[batch_id][collected] = all_permutations[batch_id][ind]
                        new_next_words[batch_id][collected] = next_words[batch_id][ind]
                        new_best_probs[batch_id][collected] = best_probs[batch_id][ind]
                        collected += 1
                    ind += 1
            best_probs = new_best_probs

            batch_indexing = torch.arange(batch_size).reshape(-1, 1)
            states = (
                unstick_two_dimensions_together(states[0], dim_value=batch_size, dim=1),
                unstick_two_dimensions_together(states[1], dim_value=batch_size, dim=1)
            )
            states = (
                states[0][:, batch_indexing, new_best_permutation, :],
                states[1][:, batch_indexing, new_best_permutation, :]
            )
            states = (stick_two_dimensions_together(states[0], dim=1), stick_two_dimensions_together(states[1], dim=1))

            current_sequences = current_sequences[batch_indexing, new_best_permutation, :]
            current_sequences[batch_indexing, self.beam_indexing, i] = new_next_words
            prev_y = unstick_two_dimensions_together(prev_y, dim_value=batch_size, dim=0)
            prev_y[batch_indexing, self.beam_indexing, 0] = new_next_words
            prev_y = stick_two_dimensions_together(prev_y, dim=0)

        sorted_probs_args = [np.flip(np.argsort(result_prob)) for result_prob in results_probs]
        return [np.array(result)[sorted_probs_args[i]] for i, result in enumerate(results)]

    def reshape_state(self, state: Tensor) -> Tensor:
        """
        Reshapes state to copying each state for batch beam_size times.
        :param state: [D1, B, D3]
        :return: [D1, B * self.beam_size, D3]
        """
        return state.unsqueeze(2).repeat((1, 1, self.beam_size, 1)).reshape(state.shape[0], -1, state.shape[-1])

    def reshape_states(self, states: List[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """
        List of states (each element is two states stored in tuple) is reshaped by self.reshape_state.
        :param states: List[Tuple[[D1, B, D3], [D4, B, D5]]]
        :return: List[Tuple[[D1, B * self.beam_size, D3], [D4, B * self.beam_size, D5]]]
        """
        reshaped_states = []
        for state in states:
            reshaped_state = (self.reshape_state(state[0]), self.reshape_state(state[1]))
            reshaped_states.append(reshaped_state)
        return reshaped_states

    def reshape_batch(self, batch: Tensor) -> Tensor:
        """
        Repeats data for each example in batch self.beam_size times.
        :param batch: [B, D2, ...]
        :return: [B * self.beam_size, D2, ...]
        """
        new_shape = [-1] + list(batch.shape[1:])
        return batch.unsqueeze(1).repeat((1, self.beam_size, 1, 1)).reshape(new_shape)

    def reshape_batches(self, batches: List[Tensor]) -> List[Tensor]:
        """
        List of batches is reshaped by self.reshape_along_batch_dim.
        :param batches: List[[B, D2, ...]]
        :return: List[[B * self.beam_size, D2, ...]]
        """
        reshaped_batches = []
        for batch in batches:
            reshaped_batch = self.reshape_batch(batch)
            reshaped_batches.append(reshaped_batch)
        return reshaped_batches


def stick_two_dimensions_together(tensor: Tensor, dim: int) -> Tensor:
    """
    Sticks dim and dim + 1 together.
    :param tensor: [..., D_i, D_{i + 1}, ...]
    :param dim: i
    :return: [..., D_i * D_{i + 1}, ...]
    """
    new_shape_before_dim = [tensor.shape[i] for i in range(dim)]
    new_shape_after_dim = [tensor.shape[i] for i in range(dim + 2, len(tensor.shape))]
    return tensor.reshape(new_shape_before_dim + [-1] + new_shape_after_dim)


def unstick_two_dimensions_together(tensor: Tensor, dim_value: int, dim: int) -> Tensor:
    """
    Sticks dim and dim + 1 together.
    :param tensor: [..., dim_value * D_{i + 1}, ...]
    :param dim_value: dimension value after unsticking
    :param dim: i
    :return: [..., dim_value, D_{i + 1}, ...]
    """
    new_shape_before_dim = [tensor.shape[i] for i in range(dim)]
    new_shape_after_dim = [tensor.shape[i] for i in range(dim + 1, len(tensor.shape))]
    return tensor.reshape(new_shape_before_dim + [dim_value, -1] + new_shape_after_dim)

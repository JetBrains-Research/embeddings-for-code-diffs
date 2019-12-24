import math
from typing import List, Tuple

import torch
import tqdm

from neural_editor.seq2seq import EncoderDecoder, Batch


class Search(object):
    """
    Class for search algorithms
    Basically user needs to feed log_probs and perform a step several times
    Results can be found in hypotheses
    """

    def __init__(self, eos_ids: List[int], vocab_size: int, search_size: int):
        self._eos_ids = eos_ids
        self._search_size = search_size
        self._vocab_size = vocab_size

    def step(self, log_probs: torch.Tensor, possible_infs: bool = False) -> torch.Tensor:
        """Take a single search step.
        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            possible_infs: whether log_probs can contain -inf
        Return:
            beams: (batch_size,)
                the hypothesis ids of the chosen elements, in the range [0, batch_size)
        """
        raise NotImplementedError

    def _step_check(self, log_probs: torch.Tensor):
        assert log_probs.size() == (
            self.batch_size,
            self._vocab_size,
        ), f"log_probs must have shape {(self.batch_size, self._vocab_size)}, but {log_probs.size()} was given"

        assert all(
            eos < self._vocab_size for eos in self._eos_ids
        ), f"EOS ids must be less than vocab_size, but EOS ids: {self._eos_ids} and vocab_size: {self._vocab_size}"

    @property
    def hypotheses(self) -> List[List[Tuple[torch.Tensor, float]]]:
        """List of list of tuples of terminated hypotheses and theirs scores"""
        raise NotImplementedError

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        raise NotImplementedError


class BeamSearch(Search):
    """Beam search algorithm with normalized by length scores"""

    def __init__(self, eos_ids: List[int], vocab_size: int, beam_size: int):
        super().__init__(eos_ids, vocab_size, beam_size)

        self._length = 1.0
        self._scores = None
        self._hypotheses = None
        self._terminated_hypotheses = []
        self._sort_mask = None
        self._eos_tensor = None

    def _init_state(self, dtype: torch.dtype, device: torch.device):
        self._device = device
        self._scores = torch.zeros(1, dtype=dtype, device=device)
        self._hypotheses = torch.empty(1, 0, dtype=torch.long, device=device)
        self._eos_tensor = torch.tensor(self._eos_ids, dtype=torch.long, device=device).unsqueeze(1)

    def step(self, log_probs: torch.Tensor, possible_infs: bool = False) -> torch.Tensor:
        """Take a single search step.
        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            possible_infs: whether log_probs can contain -inf
        Return:
            beams: (batch_size,)
                the hypothesis ids of the chosen elements, in the range [0, batch_size)
        """
        super()._step_check(log_probs)
        if self._scores is None:
            assert self._hypotheses is None
            self._init_state(log_probs.dtype, log_probs.device)

        log_probs.add_(self._scores.unsqueeze(1))
        log_probs = log_probs.flatten()
        sample_scores, samples = torch.topk(
            log_probs,
            # Take more to ensure that we will keep search_size not terminated
            min((1 + len(self._eos_ids)) * self._search_size,
                (log_probs != -math.inf).sum().item() if possible_infs else log_probs.size(0))
        )

        sort_mask = torch.div(samples, self._vocab_size)
        samples.fmod_(self._vocab_size)

        self._init_sort_mask()
        self._update_state(samples, sample_scores, sort_mask)
        self._length += 1

        return self._sort_mask

    @property
    def hypotheses(self) -> List[List[Tuple[torch.Tensor, float]]]:
        """List of tuples of terminated hypotheses and theirs scores"""
        hypotheses = [(hyp, score) for hyp, score in zip(self._hypotheses, self._scores / self._length)]
        hypotheses += self._terminated_hypotheses
        return [sorted(hypotheses, key=lambda x: x[1], reverse=True)]

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        assert (
                self._hypotheses is not None and self._hypotheses.size(1) > 0
        ), f"Can't get last predictions if no steps have been performed"
        return self._hypotheses[:, -1]

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        if self._scores is None:
            return 1
        return self._scores.size(0)

    def _init_sort_mask(self):
        self._sort_mask = torch.arange(self.batch_size)

    def _update_state(self, samples: torch.Tensor, sample_scores: torch.Tensor, sort_mask: torch.Tensor):
        self._sort_state(sort_mask)

        self._scores = sample_scores
        self._hypotheses = torch.cat((self._hypotheses, samples.unsqueeze(1)), dim=1)
        self._stash_terminated(samples)

    def _stash_terminated(self, samples: torch.Tensor):
        # We want to stash tokens only from the first search_size
        to_stash = self._is_sample_terminates(samples[:self._search_size])

        scores = self._scores / self._length
        for terminated_hypothesis, score in zip(
                self._hypotheses[: self._search_size][to_stash], scores[: self._search_size][to_stash]
        ):
            assert len(terminated_hypothesis) == int(self._length)
            self._terminated_hypotheses.append((terminated_hypothesis.clone(), score.item()))

        # And throw out all terminated
        terminated = self._is_sample_terminates(samples)
        self._apply_slice_to_state(~terminated)
        self._sort_state()

    def _sort_state(self, sort_mask: torch.Tensor = None):
        if sort_mask is None:
            _, sort_mask = torch.topk(self._scores, min(self._search_size, self._scores.size(0)))
        self._apply_slice_to_state(sort_mask)

    def _is_sample_terminates(self, samples: torch.Tensor):
        result = samples == self._eos_tensor.expand(self._eos_tensor.size(0), samples.size(0))
        return result.sum(dim=0, dtype=torch.uint8)

    def _apply_slice_to_state(self, tensor_slice):
        self._scores = self._scores[tensor_slice]
        self._hypotheses = self._hypotheses[tensor_slice]
        if self._sort_mask is not None:
            self._sort_mask = self._sort_mask[tensor_slice]


class DiverseBeamSearch(Search):
    """Beam search with diverse Hamming reward"""

    def __init__(
            self,
            eos_ids: List[int],
            vocab_size: int,
            search_size: int,
            num_groups: int,
            diversity_strength: float,
    ):
        super().__init__(eos_ids, vocab_size, search_size)

        self._num_groups = num_groups
        self._diversity_strength = -diversity_strength
        self._diversity_reward = None

        self._searches = [BeamSearch(eos_ids, vocab_size, search_size) for _ in range(num_groups)]

    def _init_diversity_reward(self, dtype: torch.dtype, device: torch.device):
        if self._diversity_reward is None:
            self._diversity_reward = torch.zeros(1, self._vocab_size, dtype=dtype, device=device)
        else:
            self._diversity_reward[:] = 0.0

    def step(self, log_probs: torch.Tensor, possible_infs: bool = False) -> torch.Tensor:
        """Take a single search step.
        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            possible_infs: whether log_probs can contain -inf
        Return:
            beams: (batch_size,)
                the hypothesis ids of the chosen elements, in the range [0, batch_size)
        """
        super()._step_check(log_probs)
        self._init_diversity_reward(log_probs.dtype, log_probs.device)

        offset = 0
        beams_sort = []
        for beam in self._searches:
            old_batch_size = beam.batch_size

            cur_log_probs = log_probs[offset: offset + old_batch_size]
            cur_beams_sort = beam.step(cur_log_probs, possible_infs)
            beams_sort.append(cur_beams_sort + offset)

            # update diversity penalty
            self._diversity_reward.scatter_add_(
                1, beam.last_predictions.unsqueeze(0), self._diversity_reward.new_ones(1, beam.batch_size)
            )
            log_probs = torch.add(log_probs, self._diversity_strength, self._diversity_reward)

            offset += old_batch_size

        return torch.cat(beams_sort)

    @property
    def hypotheses(self) -> List[List[Tuple[torch.Tensor, float]]]:
        """List of groups of hypotheses, where group is a list of tuples of terminated hypotheses and theirs scores"""
        return [beam.hypotheses[0] for beam in self._searches]

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        return torch.cat([beam.last_predictions for beam in self._searches])

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        return sum(beam.batch_size for beam in self._searches)


def perform_search(
        model: EncoderDecoder,
        batch: Batch,
        num_iterations: int,
        sos_index: int,
        terminal_id: List[int],
        beam_size: int,
        num_groups: int,
        diversity_strength: float,
        verbose: bool = False
) -> List[List[Tuple[torch.Tensor, float]]]:
    """
    :param model: trained EncoderDecoder model
    :param batch: batch with size 1
    :param num_iterations: how many iterations should perform
    :param sos_index: index of start of sentence token
    :param terminal_id: list of tokens, on which hypotheses will terminate
    :param verbose: whether to show progress bar
    :param beam_size: beam width, num performing hypotheses in each group
    :param num_groups: num diversity groups
    :param diversity_strength: how strong will be penalty for same tokens between groups
    :returns list of diversity groups, where group is list of hypotheses and their scores
    """
    src, src_mask, src_lengths = batch.src, batch.src_mask, batch.src_lengths
    with torch.no_grad():
        edit_final, encoder_output, encoder_final = model.encode(batch)
        prev_y = torch.ones(batch.nseqs, 1).fill_(sos_index).type_as(src)  # [B, 1]
        trg_mask = torch.ones_like(prev_y)  # [B, 1]

    states = None
    with torch.no_grad():
        # pre_output: [B, TrgSeqLen, DecoderH]
        out, states, pre_output = model.decode(edit_final, encoder_output, encoder_final,
                                               src_mask, prev_y, trg_mask, states)

        # we predict from the pre-output layer, which is
        # a combination of Decoder state, prev emb, and context
        log_probs = model.generator(pre_output[:, -1])  # [B, V]

    assert (
            log_probs.ndimension() == 2 and log_probs.size(0) == 1
    ), f"log_probs must have shape (1, vocab_size), but {log_probs.size()} was given"

    if verbose:
        print("----------Search info----------")
        print(f"Vocab size: {log_probs.size(1)}")
        print(f"Terminal ids: {terminal_id}")
        print(f"Num iterations: {num_iterations}")
        print(f"Beam_size: {beam_size}")
        print(f"Num diversity groups: {num_groups}")
        print(f"Diversity strength {diversity_strength}")

    if num_groups > 1:
        if verbose:
            print("Using Diverse search")
        search = DiverseBeamSearch(
            eos_ids=terminal_id,
            vocab_size=log_probs.size(1),
            search_size=beam_size,
            num_groups=num_groups,
            diversity_strength=diversity_strength,
        )
    else:
        if verbose:
            print("Using Beam search")
        search = BeamSearch(terminal_id, log_probs.size(1), beam_size)

    if verbose:
        print("-------------------------------")

    # expand batch
    log_probs = log_probs.repeat_interleave(search.batch_size, dim=0)
    src_mask = src_mask.repeat_interleave(search.batch_size * beam_size, dim=0)
    trg_mask = trg_mask.repeat_interleave(search.batch_size * beam_size, dim=0)
    edit_final = (
        edit_final[0].repeat_interleave(search.batch_size * beam_size, dim=1),
        edit_final[1].repeat_interleave(search.batch_size * beam_size, dim=1)
    )
    encoder_output = encoder_output.repeat_interleave(search.batch_size * beam_size, dim=0)
    encoder_final = (
        encoder_final[0].repeat_interleave(search.batch_size * beam_size, dim=1),
        encoder_final[1].repeat_interleave(search.batch_size * beam_size, dim=1)
    )
    states = (
        states[0].repeat_interleave(search.batch_size * beam_size, dim=1),
        states[1].repeat_interleave(search.batch_size * beam_size, dim=1)
    )

    for _ in tqdm.trange(num_iterations, disable=not verbose):
        selected_inds = search.step(log_probs, possible_infs=False)

        prev_y = search.last_predictions.unsqueeze(1)
        # pre_output: [B, TrgSeqLen, DecoderH]
        out, states, pre_output = model.decode(edit_final, encoder_output, encoder_final,
                                               src_mask, prev_y, trg_mask, states)

        # we predict from the pre-output layer, which is
        # a combination of Decoder state, prev emb, and context
        log_probs = model.generator(pre_output[:, -1])  # [B, V]

    return search.hypotheses


def get_sequence_with_maximum_probability(hypotheses: List[List[Tuple[torch.Tensor, float]]]) -> torch.Tensor:
    best_in_groups = [hypothesis[0] for hypothesis in hypotheses]
    return max(best_in_groups, key=lambda hypothesis: hypothesis[1])[0]

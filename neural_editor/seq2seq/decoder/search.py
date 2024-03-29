from typing import List, Tuple

import torch
import tqdm
import numpy as np
from neural_editor.seq2seq import EncoderDecoder, Batch
from neural_editor.seq2seq.decoder.BeamSearch import BeamSearch
from neural_editor.seq2seq.decoder.DiverseBeamSearch import DiverseBeamSearch


def create_decode_method(
        model: EncoderDecoder,
        num_iterations: int,
        sos_index: int,
        eos_index: int,
        unk_index: int,
        vocab_size: int,
        beam_size: int,
        num_groups: int,
        diversity_strength: float,
        verbose: bool = False
):
    def decode(batch) -> List[List[np.array]]:
        result = perform_search(model, batch, num_iterations, sos_index, [eos_index], unk_index, vocab_size,
                                beam_size, num_groups, diversity_strength, verbose)
        return [flat_map_and_sort_perform_search(result)]
    return decode


def perform_search(
        model: EncoderDecoder,
        batch: Batch,
        num_iterations: int,
        sos_index: int,
        terminal_id: List[int],
        unk_index: int,
        vocab_size: int,
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
        out, states, pre_output, p_gen, attn_probs = model.decode(batch, edit_final, encoder_output, encoder_final,
                                                                  src_mask, prev_y, trg_mask, states)

        # we predict from the pre-output layer, which is
        # a combination of Decoder state, prev emb, and context
        log_probs = model.generator((pre_output, p_gen, attn_probs), batch)[:, -1]  # [B, V]

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
        mask = search.step(log_probs, possible_infs=True).long()

        prev_y = search.last_predictions.unsqueeze(1)
        prev_y[prev_y >= vocab_size] = unk_index
        # pre_output: [B, TrgSeqLen, DecoderH]
        edit_final = (edit_final[0][:, mask, :], edit_final[1][:, mask, :])
        encoder_output = encoder_output[mask]
        encoder_final = (encoder_final[0][:, mask, :], encoder_final[1][:, mask, :])
        src_mask = src_mask[mask]
        trg_mask = trg_mask[mask]
        states = (states[0][:, mask, :], states[1][:, mask, :])
        out, states, pre_output, p_gen, attn_probs = model.decode(batch, edit_final, encoder_output, encoder_final,
                                                                  src_mask, prev_y, trg_mask, states)

        # we predict from the pre-output layer, which is
        # a combination of Decoder state, prev emb, and context
        log_probs = model.generator((pre_output, p_gen, attn_probs), batch)[:, -1]  # [B, V]

    return search.hypotheses


def flat_map_and_sort_perform_search(
        hypotheses: List[List[Tuple[torch.Tensor, float]]]) -> List[np.array]:
    flat_mapped = [answer for hypothesis in hypotheses for answer in hypothesis]
    flat_mapped_and_sorted = sorted(flat_mapped, key=lambda answer: answer[1], reverse=True)
    return list(map(lambda answer: answer[0].detach().cpu().numpy()[:-1], flat_mapped_and_sorted))


def get_sequence_with_maximum_probability(hypotheses: List[List[Tuple[torch.Tensor, float]]]) -> torch.Tensor:
    best_in_groups = [hypothesis[0] for hypothesis in hypotheses]
    return max(best_in_groups, key=lambda hypothesis: hypothesis[1])[0]


def get_shortest_sequence(hypotheses: List[List[Tuple[torch.Tensor, float]]]) -> torch.Tensor:
    all_predictions = [prediction for hypothesis in hypotheses for prediction in hypothesis]
    return min(all_predictions, key=lambda prediction: len(prediction[0]))[0]


def greedy_decode(
        model: EncoderDecoder,
        batch: Batch,
        num_iterations: int,
        sos_index: int,
        terminal_ids: List[int],
        unk_index: int,
        vocab_size: int,
        verbose: bool = False
) -> torch.Tensor:
    hypotheses = perform_search(model, batch, num_iterations, sos_index, terminal_ids, unk_index, vocab_size,
                                beam_size=1, num_groups=1, diversity_strength=None, verbose=verbose)
    return get_shortest_sequence(hypotheses)
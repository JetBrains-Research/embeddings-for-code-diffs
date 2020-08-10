from typing import List, Tuple


def diff_sequences_and_add_hunks(prev_line, updated_line, differ, config):
    diff = differ.diff_tokens_fast_lvn(prev_line.split(' '), updated_line.split(' '),
                                       leave_only_changed=False)
    diff_hunk_ids, prev_hunk_ids, updated_hunk_ids = split_on_hunks(diff,
                                                                    config['UNCHANGED_TOKEN'],
                                                                    config['PADDING_TOKEN'])
    if config['LEAVE_ONLY_CHANGED']:
        diff, diff_hunk_ids = leave_only_changed(diff, diff_hunk_ids, config['UNCHANGED_TOKEN'])
    prev_line = ' '.join(insert_new_hunk_token_into_sequence(prev_line.split(' '), prev_hunk_ids,
                                                             config['HUNK_TOKEN']))
    updated_line = ' '.join(insert_new_hunk_token_into_sequence(updated_line.split(' '), updated_hunk_ids,
                                                                config['HUNK_TOKEN']))
    diff = insert_new_hunk_token_into_diff(diff, diff_hunk_ids, config['HUNK_TOKEN'])
    return diff, prev_line, updated_line


def split_on_hunks(diff: Tuple[List[str], List[str], List[str]],
                   equality_sign: str, padding_token: str, context_size=10) -> Tuple[List[int], List[int], List[int]]:
    prev_indices = []
    updated_indices = []
    diff_indices = []

    equality_signs_num = 0
    seen_changed = False
    seen_prev_paddings_num = 0
    seen_updated_paddings_num = 0
    for idx, (alignment_sign, prev_token, updated_token) in enumerate(zip(diff[0], diff[1], diff[2])):
        if alignment_sign == equality_sign:
            equality_signs_num += 1
        else:
            seen_changed = True
            equality_signs_num = 0
        if prev_token == padding_token:
            seen_prev_paddings_num += 1
        if updated_token == padding_token:
            seen_updated_paddings_num += 1
        if equality_signs_num == context_size and seen_changed and idx + 1 != len(diff[0]):
            diff_indices.append(idx + 1)
            prev_indices.append(idx + 1 - seen_prev_paddings_num)
            updated_indices.append(idx + 1 - seen_updated_paddings_num)
            equality_signs_num = 0
            seen_changed = False
    return diff_indices, prev_indices, updated_indices


def leave_only_changed(diff: Tuple[List[str], List[str], List[str]], diff_hunk_indices: List[int],
                       equality_sign: str) -> Tuple[Tuple[List[str], List[str], List[str]], List[int]]:
    diff_alignment, diff_prev, diff_updated = [], [], []
    new_hunk_indices = []
    cur_idx = 0
    for idx, (alignment_token, prev_token, updated_token) in enumerate(zip(diff[0], diff[1], diff[2])):
        if alignment_token != equality_sign:
            diff_alignment.append(alignment_token)
            diff_prev.append(prev_token)
            diff_updated.append(updated_token)
        if cur_idx < len(diff_hunk_indices) and idx == diff_hunk_indices[cur_idx] - 1:
            new_hunk_indices.append(len(diff_alignment))
            cur_idx += 1
    return (diff_alignment, diff_prev, diff_updated), new_hunk_indices


def insert_new_hunk_token_into_sequence(sequence: List[str], sequence_hunk_ids: List[int], new_hunk_token: str) -> List[str]:
    result_sequence = [new_hunk_token]
    cur_idx = 0
    for i in range(len(sequence)):
        if cur_idx < len(sequence_hunk_ids) and i == sequence_hunk_ids[cur_idx]:
            result_sequence.append(new_hunk_token)
            cur_idx += 1
        result_sequence.append(sequence[i])
    # because of removing unchanged tokens sometimes hunk tokens should be inserted in the end
    while cur_idx < len(sequence_hunk_ids):
        result_sequence.append(new_hunk_token)
        cur_idx += 1
    return result_sequence


def insert_new_hunk_token_into_diff(diff: Tuple[List[str], List[str], List[str]], diff_hunk_ids: List[int],
                                    new_hunk_token: str) -> Tuple[List[str], List[str], List[str]]:
    return (
        insert_new_hunk_token_into_sequence(diff[0], diff_hunk_ids, new_hunk_token),
        insert_new_hunk_token_into_sequence(diff[1], diff_hunk_ids, new_hunk_token),
        insert_new_hunk_token_into_sequence(diff[2], diff_hunk_ids, new_hunk_token),
    )

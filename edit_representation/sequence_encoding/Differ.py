from typing import List, Dict, Tuple

import Levenshtein as Lvn

# TODO: write unit test with example from vk, problem: '' appears in output


def build_vocab(tokens: List[str]) -> Dict[str, chr]:
    max_unseen = 0
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = chr(max_unseen)
            max_unseen += 1
    return vocab


def to_ids_str(prev: List[str], vocab: Dict[str, chr]) -> str:
    return ''.join(map(lambda token: str(vocab[token]), prev))


class Differ:
    def __init__(self, replacement_token: str, deletion_token: str,
                 addition_token: str, unchanged_token: str, padding_token: str) -> None:
        super().__init__()
        self.replacement_token = replacement_token
        self.deletion_token = deletion_token
        self.addition_token = addition_token
        self.equal_token = unchanged_token
        self.padding_token = padding_token

    def create_aligned_sequences(self, prev: List[str], updated: List[str],
                                 opcodes: List[Tuple[str, int, int, int, int]])\
            -> Tuple[List[str], List[str], List[str]]:
        operations = []
        prev_result = []
        updated_result = []
        for opcode in opcodes:
            if opcode[0] == 'delete':
                # assert(len(range(opcode[1], opcode[2])) > 0)
                # assert(len(range(opcode[3], opcode[4])) == 0)
                operations += [self.deletion_token for i in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [self.padding_token for i in range(opcode[1], opcode[2])]
            elif opcode[0] == 'insert':
                # assert(len(range(opcode[1], opcode[2])) == 0)
                # assert(len(range(opcode[3], opcode[4])) > 0)
                operations += [self.addition_token for _ in range(opcode[3], opcode[4])]
                prev_result += [self.padding_token for _ in range(opcode[3], opcode[4])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
            elif opcode[0] == 'equal':
                # assert(len(range(opcode[1], opcode[2])) == len(range(opcode[3], opcode[4])))
                operations += [self.equal_token for _ in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
            elif opcode[0] == 'replace':
                # assert(len(range(opcode[1], opcode[2])) == len(range(opcode[3], opcode[4])))
                operations += [self.replacement_token for _ in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
        return operations, prev_result, updated_result

    def diff_tokens_fast_lvn(self, prev: List[str], updated: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        :param prev: previous token sequence
        :param updated: updated token sequence
        :return: aligned sequences as triple (operations, prev aligned, updated aligned)
        """
        vocab = build_vocab(prev + updated)
        prev_ids = to_ids_str(prev, vocab)
        updated_ids = to_ids_str(updated, vocab)
        opcodes = Lvn.opcodes(prev_ids, updated_ids)
        return self.create_aligned_sequences(prev, updated, opcodes)


if __name__ == "__main__":
    prev_tokens = ['v', '.', 'F', '=', 'x', '+', 'x']
    updated_tokens = ['u', '=', 'x', '+', 'x', ';']
    diff = Differ('↔', '−', '+', '=', '∅').diff_tokens_fast_lvn(prev_tokens, updated_tokens)
    for line in diff:
        print(line)

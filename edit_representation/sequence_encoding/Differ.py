import json

import torch
from collatex import *

from neural_editor.seq2seq.train_config import CONFIG


class Differ:
    # TODO: is for loop on batches bottleneck?

    def create_aligned_sequences(self, prev, updated, opcodes):
        operations = []
        prev_result = []
        updated_result = []
        for opcode in opcodes:
            if opcode[0] == 'delete':
                assert(len(range(opcode[1], opcode[2])) > 0)
                assert(len(range(opcode[3], opcode[4])) == 0)
                operations += [self.deletion_code for i in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [self.padding_code for i in range(opcode[1], opcode[2])]
            elif opcode[0] == 'insert':
                assert(len(range(opcode[1], opcode[2])) == 0)
                assert(len(range(opcode[3], opcode[4])) > 0)
                operations += [self.addition_code for _ in range(opcode[3], opcode[4])]
                prev_result += [self.padding_code for _ in range(opcode[3], opcode[4])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
            elif opcode[0] == 'equal':
                assert(len(range(opcode[1], opcode[2])) == len(range(opcode[3], opcode[4])))
                operations += [self.equal_code for _ in range(opcode[1], opcode[2])]
                try:
                    prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                except:
                    a = 0
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
            elif opcode[0] == 'replace':
                assert(len(range(opcode[1], opcode[2])) == len(range(opcode[3], opcode[4])))
                operations += [self.replacement_code for _ in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
        return [operations, prev_result, updated_result]

    def __init__(self, replacement_code, deletion_code, addition_code, unchanged_code, padding_code) -> None:
        super().__init__()
        self.replacement_code = replacement_code
        self.deletion_code = deletion_code
        self.addition_code = addition_code
        self.equal_code = unchanged_code
        self.padding_code = padding_code

    def diff_token_tensors(self, prev_batches, updated_batches):
        result = []
        for i in range(prev_batches.size()[0]):
            result.append(self.diff_token_ids_int(prev_batches[i], updated_batches[i]))
        return torch.tensor(result)

    def diff_token_ids_int(self, prev, updated):
        prev = [str(int(prev[i])) for i in range(prev.size()[0])]
        updated = [str(int(updated[i])) for i in range(updated.size()[0])]
        return self.diff_token_ids(prev, updated)

    def diff_token_ids(self, prev, updated):
        tokens_to_align = {
            'witnesses': [
                {
                    'id': 'prev',
                    'tokens': [{'t': token, 'n': token} for token in prev]
                },
                {
                    'id': 'updated',
                    'tokens': [{'t': token, 'n': token} for token in updated]
                },
            ]
        }
        alignment_table = collate(tokens_to_align, near_match=True, segmentation=False)
        alignment = [[], [], []]
        for column in alignment_table.columns:
            diff_dict = column.tokens_per_witness
            if 'prev' in diff_dict.keys() and 'updated' in diff_dict.keys():
                prev_token = int(str(diff_dict['prev'][0]))
                updated_token = int(str(diff_dict['updated'][0]))
                if prev_token == updated_token:
                    alignment[0].append(self.equal_code)
                else:
                    alignment[0].append(self.replacement_code)
                alignment[1].append(prev_token)
                alignment[2].append(updated_token)
            elif 'prev' in diff_dict.keys():
                alignment[0].append(self.deletion_code)
                alignment[1].append(int(str(diff_dict['prev'][0])))
                alignment[2].append(self.padding_code)
            elif 'updated' in diff_dict.keys():
                alignment[0].append(self.addition_code)
                alignment[1].append(self.padding_code)
                alignment[2].append(int(str(diff_dict['updated'][0])))
        return alignment

    def build_diffs_vocab(self, data):
        pass

    def build_vocab(self, tokens):
        max_unseen = 0
        vocab = {}
        for token in tokens:
            if token not in vocab:
                vocab[token] = chr(max_unseen)
                max_unseen += 1
        return vocab

    def diff_tokens_bio_pairwise(self, prev, updated):
        from Bio import pairwise2
        vocab = self.build_vocab(prev + updated)
        prev_ids = self.to_ids_str(prev, vocab)
        updated_ids = self.to_ids_str(updated, vocab)
        alignments = pairwise2.align.globalxx(prev_ids, updated_ids)

    def diff_tokens_fast_lvn(self, prev, updated):
        import Levenshtein as lvn
        vocab = self.build_vocab(prev + updated)
        prev_ids = self.to_ids_str(prev, vocab)
        updated_ids = self.to_ids_str(updated, vocab)
        opcodes = lvn.opcodes(prev_ids, updated_ids)
        return self.create_aligned_sequences(prev, updated, opcodes)

    def diff_tokens(self, prev, updated):
        tokens_to_align = {
            'witnesses': [
                {
                    'id': 'prev',
                    'tokens': [{'t': token, 'n': token} for token in prev]
                },
                {
                    'id': 'updated',
                    'tokens': [{'t': token, 'n': token} for token in updated]
                },
            ]
        }
        alignment_table = collate(tokens_to_align, near_match=True, segmentation=False)
        alignment = [[], [], []]
        for column in alignment_table.columns:
            diff_dict = column.tokens_per_witness
            if 'prev' in diff_dict.keys() and 'updated' in diff_dict.keys():
                prev_token = str(diff_dict['prev'][0])
                updated_token = str(diff_dict['updated'][0])
                if prev_token == updated_token:
                    alignment[0].append(self.equal_code)
                else:
                    alignment[0].append(self.replacement_code)
                alignment[1].append(prev_token)
                alignment[2].append(updated_token)
            elif 'prev' in diff_dict.keys():
                alignment[0].append(self.deletion_code)
                alignment[1].append(str(diff_dict['prev'][0]))
                alignment[2].append(self.padding_code)
            elif 'updated' in diff_dict.keys():
                alignment[0].append(self.addition_code)
                alignment[1].append(self.padding_code)
                alignment[2].append(str(diff_dict['updated'][0]))
        return alignment

    def to_ids_str(self, prev, vocab):
        return ''.join(map(lambda token: str(vocab[token]), prev))


if __name__ == "__main__":
    print("STRING EXAMPLE")
    prev = ['v', '.', 'F', '=', 'x', '+', 'x']
    updated = ['u', '=', 'x', '+', 'x', ';', ';']
    #diff = Differ('↔', '−', '+', '=', '∅').diff_tokens_bio_pairwise(prev, updated)
    diff = Differ('↔', '−', '+', '=', '∅').diff_tokens_fast_lvn(prev, updated)
    for line in diff:
        print(line)

    print()

    print("TENSORS EXAMPLE")
    diff = Differ(8, 9, 10, 11, 12).diff_token_tensors(
        torch.tensor([[0, 1, 2, 3, 4, 5, 4], [0, 1, 2, 3, 4, 5, 4]]),
        torch.tensor([[6, 3, 4, 5, 4, 7], [6, 3, 4, 5, 4, 7]])
    )
    print(diff.size())
    print(diff)
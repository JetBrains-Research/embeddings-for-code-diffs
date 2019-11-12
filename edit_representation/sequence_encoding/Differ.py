import Levenshtein as Lvn


def build_vocab(tokens):
    max_unseen = 0
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = chr(max_unseen)
            max_unseen += 1
    return vocab


def to_ids_str(prev, vocab):
    return ''.join(map(lambda token: str(vocab[token]), prev))


class Differ:
    def create_aligned_sequences(self, prev, updated, opcodes):
        operations = []
        prev_result = []
        updated_result = []
        for opcode in opcodes:
            if opcode[0] == 'delete':
                # assert(len(range(opcode[1], opcode[2])) > 0)
                # assert(len(range(opcode[3], opcode[4])) == 0)
                operations += [self.deletion_code for i in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [self.padding_code for i in range(opcode[1], opcode[2])]
            elif opcode[0] == 'insert':
                # assert(len(range(opcode[1], opcode[2])) == 0)
                # assert(len(range(opcode[3], opcode[4])) > 0)
                operations += [self.addition_code for _ in range(opcode[3], opcode[4])]
                prev_result += [self.padding_code for _ in range(opcode[3], opcode[4])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
            elif opcode[0] == 'equal':
                # assert(len(range(opcode[1], opcode[2])) == len(range(opcode[3], opcode[4])))
                operations += [self.equal_code for _ in range(opcode[1], opcode[2])]
                prev_result += [prev[i] for i in range(opcode[1], opcode[2])]
                updated_result += [updated[i] for i in range(opcode[3], opcode[4])]
            elif opcode[0] == 'replace':
                # assert(len(range(opcode[1], opcode[2])) == len(range(opcode[3], opcode[4])))
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

    def diff_tokens_fast_lvn(self, prev, updated):
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

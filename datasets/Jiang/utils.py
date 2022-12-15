from typing import List


def tokenize_git_diff_output_string(git_diff_output: str) -> List[List[str]]:
    tokens = git_diff_output.split()
    tokens_per_line = [[]]
    for token in tokens:
        if token == '<nl>':
            tokens_per_line.append([])
        else:
            tokens_per_line[-1].append(token)
    if len(tokens_per_line[-1]) == 0:
        tokens_per_line = tokens_per_line[:-1]
    return tokens_per_line

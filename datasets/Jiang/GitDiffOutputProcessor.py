import itertools
from typing import Tuple, List


class GitDiffOutputProcessor:
    """
    This class can be used to generate prev and updated code from output of git diff command.
    """

    @staticmethod
    def get_prev_and_updated(git_diff_output: str) -> Tuple[str, str]:
        """
        Generates prev and updated code from output of git diff command.
        :param git_diff_output: output of git diff command
        :return: [prev, updated]
        """
        tokens = git_diff_output.split()
        tokens_per_line = [[]]
        for token in tokens:
            if token == '<nl>':
                tokens_per_line.append([])
            else:
                tokens_per_line[-1].append(token)
        if len(tokens_per_line[-1]) == 0:
            tokens_per_line = tokens_per_line[:-1]

        prev_lines, updated_lines = [], []
        for tokens_in_line in tokens_per_line:
            if tokens_in_line[0] == 'mmm':
                prev_lines.append(tokens_in_line)
            elif tokens_in_line[0] == 'ppp':
                updated_lines.append(tokens_in_line)
            elif tokens_in_line[:2] == ['new', 'file']:
                prev_lines.append(tokens_in_line)
            elif tokens_in_line[0] == 'deleted':
                updated_lines.append(tokens_in_line)
            elif tokens_in_line[0] == '-':
                prev_lines.append(tokens_in_line[1:])
            elif tokens_in_line[0] == '+':
                updated_lines.append(tokens_in_line[1:])
            elif tokens_in_line[0] == 'index':
                continue
            else:
                prev_lines.append(tokens_in_line)
                updated_lines.append(tokens_in_line)

        # TODO: leave <nl> in the end?
        prev = ' '.join(itertools.chain(*[line + ['<nl>'] for line in prev_lines]))
        updated = ' '.join(itertools.chain(*[line + ['<nl>'] for line in updated_lines]))

        assert (prev != updated)
        return prev, updated

    @staticmethod
    def get_prev_and_updated_for_diffs(git_diff_outputs: List[str]) -> List[Tuple[str, str]]:
        result = []
        for git_diff_output in git_diff_outputs:
            prev_and_updated = GitDiffOutputProcessor.get_prev_and_updated(git_diff_output)
            if prev_and_updated is not None:
                result.append(prev_and_updated)
        return result


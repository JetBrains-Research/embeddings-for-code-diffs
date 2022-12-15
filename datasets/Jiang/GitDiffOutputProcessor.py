import itertools
from typing import Tuple, List

from datasets.Jiang.utils import tokenize_git_diff_output_string


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
        tokens_per_line = tokenize_git_diff_output_string(git_diff_output)

        prev_lines, updated_lines = [], []
        was_special_keyword_modification = False
        for tokens_in_line in tokens_per_line:
            if tokens_in_line[0] == 'mmm':
                prev_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[0] == 'ppp':
                updated_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:3] == ['new', 'file', 'mode']:
                prev_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:3] == ['deleted', 'file', 'mode']:
                updated_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['rename', 'from']:
                prev_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['rename', 'to']:
                updated_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['old', 'mode']:
                prev_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['new', 'mode']:
                updated_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[0] == '-':
                prev_lines.append(tokens_in_line[1:])
            elif tokens_in_line[0] == '+':
                updated_lines.append(tokens_in_line[1:])
            elif tokens_in_line[0] == 'index' or tokens_in_line[:2] == ['similarity', 'index']:
                continue
            else:
                prev_lines.append(tokens_in_line)
                updated_lines.append(tokens_in_line)

        # TODO: leave <nl> in the end?
        prev = ' '.join(itertools.chain(*[line + ['<nl>'] for line in prev_lines]))
        updated = ' '.join(itertools.chain(*[line + ['<nl>'] for line in updated_lines]))

        if not was_special_keyword_modification:
            print(f'No special keyword found for diff: {git_diff_output}')
        if prev == updated:
            print(f'Prev and updated are the same for diff: {git_diff_output}')
        return prev, updated

    @staticmethod
    def get_prev_and_updated_for_diffs(git_diff_outputs: List[str]) -> List[Tuple[str, str]]:
        result = []
        for git_diff_output in git_diff_outputs:
            prev_and_updated = GitDiffOutputProcessor.get_prev_and_updated(git_diff_output)
            if prev_and_updated is not None:
                result.append(prev_and_updated)
        return result


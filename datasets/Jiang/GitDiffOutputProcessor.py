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

        #if tokens_per_line[0][0] == 'mmm':
        #    prev_lines, updated_lines = GitDiffOutputProcessor.process_modification(tokens_per_line)
        #elif tokens_per_line[0][0] == 'deleted':
        #    prev_lines, updated_lines = GitDiffOutputProcessor.process_file_deletion(tokens_per_line)
        #elif tokens_per_line[0][0] == 'new' and tokens_per_line[0][1] == 'file':
        #    prev_lines, updated_lines = GitDiffOutputProcessor.process_file_addition(tokens_per_line)
        #else:
        #    raise Exception(f'Illegal diff:\n{git_diff_output}')

        #prev_lines, updated_lines = \
        #    GitDiffOutputProcessor.handle_additions_and_deletions(prev_lines, updated_lines)
        # TODO: leave <nl> in the end?
        prev = ' '.join(itertools.chain(*[line + ['<nl>'] for line in prev_lines]))
        updated = ' '.join(itertools.chain(*[line + ['<nl>'] for line in updated_lines]))

        assert (prev != updated)
        return prev, updated

    @staticmethod
    def handle_additions_and_deletions(prev_lines: List[List[str]],
                                       updated_lines: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        return GitDiffOutputProcessor.handle_additions_and_deletions_prev_specified(prev_lines, is_prev=True), \
               GitDiffOutputProcessor.handle_additions_and_deletions_prev_specified(updated_lines, is_prev=False)

    @staticmethod
    def handle_additions_and_deletions_prev_specified(tokens_per_line: List[List[str]], is_prev: bool) -> List[List[str]]:
        sign_to_delete = '-' if is_prev else '+'
        line_to_delete = ['ppp', '+'] if is_prev else ['mmm', '-']
        return list(map(lambda x: x[1:] if x[0] == sign_to_delete else x,
                        filter(lambda x: not x[0] in line_to_delete, tokens_per_line)))

    @staticmethod
    def get_prev_and_updated_for_diffs(git_diff_outputs: List[str]) -> List[Tuple[str, str]]:
        result = []
        for git_diff_output in git_diff_outputs:
            prev_and_updated = GitDiffOutputProcessor.get_prev_and_updated(git_diff_output)
            if prev_and_updated is not None:
                result.append(prev_and_updated)
        return result

    @staticmethod
    def process_modification(tokens_per_line: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        assert (tokens_per_line[0][0] == 'mmm')
        assert (tokens_per_line[1][0] == 'ppp')

        prev_lines = [tokens_per_line[0]] + tokens_per_line[2:]
        updated_lines = [tokens_per_line[1]] + tokens_per_line[2:]
        return prev_lines, updated_lines

    @staticmethod
    def process_file_deletion(tokens_per_line: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        start_index = 1
        if tokens_per_line[1][0] == 'index':
            start_index += 1
        return tokens_per_line[start_index:], [tokens_per_line[0]] + tokens_per_line[start_index:]

    @staticmethod
    def process_file_addition(tokens_per_line: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        start_index = 1
        if tokens_per_line[1][0] == 'index':
            start_index += 1
        return [tokens_per_line[0]] + tokens_per_line[start_index:], tokens_per_line[start_index:]

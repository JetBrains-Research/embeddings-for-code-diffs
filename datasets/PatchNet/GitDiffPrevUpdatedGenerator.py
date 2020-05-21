from typing import Dict, List

import pydriller

from datasets.PatchNet.tokenizers import PygmentsCTokenizer


class SimpleGitDiffProcessor:
    """
    Simple git diff is split on prev and updated: '-' goes to prev, '+' goes to updated. Nothing else.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = PygmentsCTokenizer()

    def get_prev_and_updated(self, diff_output: str) -> Dict[str, List[str]]:
        prev_tokens, updated_tokens = [], []
        prev_lines, updated_lines = [], []
        diff_lines = diff_output.splitlines(keepends=True)
        for i, diff_line in enumerate(diff_lines):
            if self.is_new_hunk(diff_line) and i != 0:
                prev_tokens += self.tokenizer.tokenize(''.join(prev_lines))
                updated_tokens += self.tokenizer.tokenize(''.join(updated_lines))
                prev_lines, updated_lines = [], []
            if self.is_common_line(diff_line):
                processed_line = self.process_common_line(diff_line)
                prev_lines.append(processed_line)
                updated_lines.append(processed_line)
            elif self.is_only_prev_line(diff_line):
                processed_line = self.process_only_prev_line(diff_line)
                prev_lines.append(processed_line)
            elif self.is_only_updated_line(diff_line):
                processed_line = self.process_only_updated_line(diff_line)
                updated_lines.append(processed_line)
        prev_tokens += self.tokenizer.tokenize(''.join(prev_lines))
        updated_tokens += self.tokenizer.tokenize(''.join(updated_lines))
        return {'prev': prev_tokens, 'updated': updated_tokens}

    def is_common_line(self, diff_line: str) -> bool:
        return diff_line.startswith(' ') or diff_line.startswith('@@')

    def process_common_line(self, diff_line: str) -> str:
        if diff_line.startswith(' '):
            return diff_line[1:]
        elif diff_line.startswith('@@'):
            return diff_line.split('@@')[-1][1:]
        else:
            raise Exception(f'Passed line {diff_line} is not a common diff line')

    def is_only_prev_line(self, diff_line: str) -> bool:
        return diff_line.startswith('-')

    def process_only_prev_line(self, diff_line: str) -> str:
        if diff_line.startswith('-'):
            return diff_line[1:]
        else:
            raise Exception(f'Passed line {diff_line} is not an only prev diff line')

    def is_only_updated_line(self, diff_line: str) -> bool:
        return diff_line.startswith('+')

    def process_only_updated_line(self, diff_line: str) -> str:
        if diff_line.startswith('+'):
            return diff_line[1:]
        else:
            raise Exception(f'Passed line {diff_line} is not an only updated diff line')

    def is_new_hunk(self, diff_line: str) -> bool:
        return diff_line.startswith('@@')


class GitDiffPrevUpdatedGenerator:
    def __init__(self) -> None:
        super().__init__()
        self.git_diff_processor = SimpleGitDiffProcessor()

    def generate_prev_and_updated(self, commit: pydriller.Commit) -> Dict[str, List[str]]:
        prev_tokens, updated_tokens = [], []
        for modified_file in commit.modifications:
            out = self.git_diff_processor.get_prev_and_updated(modified_file.diff)
            prev_tokens += out['prev']
            updated_tokens += out['updated']
        return {'prev': prev_tokens, 'updated': updated_tokens}

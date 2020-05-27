from collections import Counter
from typing import Dict, List, Tuple

import pydriller

from datasets.PatchNet.tokenizers import PygmentsCTokenizer


class SimpleGitDiffProcessor:
    """
    Simple git diff is split on prev and updated: '-' goes to prev, '+' goes to updated. Nothing else.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = PygmentsCTokenizer()

    def apply_tokenizer(self, lines, tokens_list, identifier_names_counter):
        tokens, counter = self.tokenizer.tokenize(''.join(lines))
        tokens_list += tokens
        identifier_names_counter += counter

    def get_prev_and_updated(self, diff_output: str) -> Tuple[Dict[str, List[str]], Counter]:
        prev_tokens, updated_tokens = [], []
        prev_lines, updated_lines = [], []
        diff_lines = diff_output.splitlines(keepends=True)
        identifier_names_counter = Counter()
        for i, diff_line in enumerate(diff_lines):
            if self.is_new_hunk(diff_line) and i != 0:
                self.apply_tokenizer(prev_lines, prev_tokens, identifier_names_counter)
                self.apply_tokenizer(updated_lines, updated_tokens, identifier_names_counter)
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
        self.apply_tokenizer(prev_lines, prev_tokens, identifier_names_counter)
        self.apply_tokenizer(updated_lines, updated_tokens, identifier_names_counter)
        return {'prev': prev_tokens, 'updated': updated_tokens}, identifier_names_counter

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

    def generate_prev_and_updated(self, commit: pydriller.Commit) -> Tuple[Dict[str, List[str]], Counter]:
        prev_tokens, updated_tokens = [], []
        identifier_names_counter = Counter()
        for modified_file in commit.modifications:
            prev_updated, counter = self.git_diff_processor.get_prev_and_updated(modified_file.diff)
            prev_tokens += prev_updated['prev']
            updated_tokens += prev_updated['updated']
            identifier_names_counter += counter
        return {'prev': prev_tokens, 'updated': updated_tokens}, identifier_names_counter

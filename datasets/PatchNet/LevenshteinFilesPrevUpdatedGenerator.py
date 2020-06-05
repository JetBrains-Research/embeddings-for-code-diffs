from collections import Counter
from typing import Dict, List, Tuple
import pydriller
from pydriller import ModificationType

from datasets.PatchNet.tokenizers import PygmentsCTokenizer
from edit_representation.sequence_encoding.Differ import Differ


class FilesDiffProcessor:
    REPLACEMENT_TOKEN = 'замена'
    DELETION_TOKEN = 'удаление'
    ADDITION_TOKEN = 'добавление'
    UNCHANGED_TOKEN = 'равенство'
    PADDING_TOKEN = 'паддинг'

    def __init__(self, context_size=10) -> None:
        super().__init__()
        self.tokenizer = PygmentsCTokenizer()
        self.context_size = context_size
        self.differ = Differ('замена', 'удаление', 'добавление', 'равенство', 'паддинг')

    def get_prev_and_updated(self, file_before: str, file_after: str) -> Tuple[Dict[str, List[str]], Counter]:
        prev_tokens = self.tokenizer.tokenize_with_types(file_before)
        updated_tokens = self.tokenizer.tokenize_with_types(file_after)
        cut_prev_tokens, cut_updated_tokens = self.cut_sequences(prev_tokens, updated_tokens)
        identifier_names_counter = PygmentsCTokenizer.count_identifier_names(cut_prev_tokens + cut_updated_tokens)
        return {'prev': [t[1] for t in cut_prev_tokens], 'updated': [t[1] for t in cut_updated_tokens]}, \
               identifier_names_counter

    def cut_sequences(self, prev_tokens, updated_tokens):
        cut_ranges = self.get_ranges_to_extract([t[1] for t in prev_tokens], [t[1] for t in updated_tokens])
        cut_prev_tokens, cut_updated_tokens = [], []
        for cut_range in cut_ranges:
            cut_prev_tokens += prev_tokens[cut_range[0][0]:cut_range[0][1]]
            cut_updated_tokens += updated_tokens[cut_range[1][0]:cut_range[1][1]]
        return cut_prev_tokens, cut_updated_tokens

    def get_ranges_to_extract(self, prev_tokens: List[str], updated_tokens: List[str]) -> List[Tuple[Tuple[int, int],
                                                                                                     Tuple[int, int]]]:
        diff_alignment, diff_prev, diff_updated = \
            self.differ.diff_tokens_fast_lvn(prev_tokens, updated_tokens, leave_only_changed=False)
        prev_padding = 0
        updated_padding = 0
        prev_mapping = []
        updated_mapping = []
        blocks = []
        start = None

        for i in range(len(diff_alignment)):
            if start is None and diff_alignment[i] != FilesDiffProcessor.UNCHANGED_TOKEN:
                start = i
            if start is not None and diff_alignment[i] == FilesDiffProcessor.UNCHANGED_TOKEN:
                blocks.append((start, i))
                start = None
            prev_mapping.append(i - prev_padding)
            updated_mapping.append(i - updated_padding)
            if diff_prev[i] == FilesDiffProcessor.PADDING_TOKEN:
                prev_padding += 1
            if diff_updated[i] == FilesDiffProcessor.PADDING_TOKEN:
                updated_padding += 1
        if start is not None:
            blocks.append((start, len(diff_alignment)))
            start = None
        prev_mapping.append(len(prev_tokens))
        updated_mapping.append(len(updated_tokens))
        result = []
        for i in range(len(blocks)):
            if i == 0:
                left = max(0, blocks[i][0] - self.context_size)
            else:
                left = max(right, blocks[i][0] - self.context_size)
            if i + 1 == len(blocks):
                right = min(len(diff_alignment), blocks[i][1] + self.context_size)
            else:
                right = min(blocks[i + 1][0], blocks[i][1] + self.context_size)
            result.append(((prev_mapping[left], prev_mapping[right]),
                           (updated_mapping[left], updated_mapping[right])))
        return result


class LevenshteinFilesPrevUpdatedGenerator:
    def __init__(self) -> None:
        super().__init__()
        self.files_diff_processor = FilesDiffProcessor()

    @staticmethod
    def extract_prev_updated_files(commit) -> List[Tuple[str, str]]:
        return [(f.source_code_before, f.source_code) for f in commit.modifications
                if f.change_type == ModificationType.MODIFY]

    @staticmethod
    def extract_prev_updated_methods(commit) -> List[Tuple[str, str]]:
        result = []
        for modified_file in commit.modifications:
            lines_before = modified_file.source_code_before.splitlines(keepends=True)
            lines_after = modified_file.source_code.splitlines(keepends=True)
            for changed_method in modified_file.changed_methods:
                found = False
                for method_before in modified_file.methods_before:
                    if changed_method == method_before:
                        prev = ''.join(lines_before[method_before.start_line - 1:method_before.end_line])
                        updated = ''.join(lines_after[changed_method.start_line - 1:changed_method.end_line])
                        result.append((prev, updated))
                        found = True
                        break
                if not found:
                    print(f'No pair method found for commit with hash {commit.hash}')
        if len(result) == 0:
            print(f'No changed methods are found for commit with hash {commit.hash}')
        return result

    def generate_prev_and_updated(self, commit: pydriller.Commit) -> Tuple[Dict[str, List[str]], Counter]:
        prev_tokens, updated_tokens = [], []
        identifier_names_counter = Counter()
        try:
            for prev, updated in LevenshteinFilesPrevUpdatedGenerator.extract_prev_updated_files(commit):
                prev_updated, counter = self.files_diff_processor.get_prev_and_updated(prev, updated)
                prev_tokens += prev_updated['prev']
                updated_tokens += prev_updated['updated']
                identifier_names_counter += counter
            if len(prev_tokens) == 0 or len(updated_tokens) == 0:
                print(f'No changes found for commit with hash {commit.hash}')
        except Exception as e:
            print(f'Error for commit {commit.hash}: {e}, {e.__class__}')
        return {'prev': prev_tokens, 'updated': updated_tokens}, identifier_names_counter

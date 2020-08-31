import unittest

from datasets.HunkSplitter import HunkSplitter
from edit_representation.sequence_encoding.Differ import Differ


class HunkSplitterTest(unittest.TestCase):
    CONFIG = {
        'REPLACEMENT_TOKEN': 'r',
        'DELETION_TOKEN': 'd',
        'ADDITION_TOKEN': 'a',
        'UNCHANGED_TOKEN': 'u',
        'PADDING_TOKEN': 'p',
        'HUNK_TOKEN': 'h',
        'LEAVE_ONLY_CHANGED': True,
    }
    DIFFER = Differ(CONFIG['REPLACEMENT_TOKEN'], CONFIG['DELETION_TOKEN'],
                    CONFIG['ADDITION_TOKEN'], CONFIG['UNCHANGED_TOKEN'],
                    CONFIG['PADDING_TOKEN'])
    HUNK_SPLITTER = HunkSplitter(context_size=2, differ=DIFFER, config=CONFIG)

    def test_diff_sequences_and_add_hunks_on_base_testing_example(self):
        prev_line    = 'c1 t1 t2 t3 t4 c3 c4 c5 c6 c7 c8'
        updated_line = 'c1 t6 t2 t7 c3 c4 c5 c6 t5 t3 c7 c8'

        diff_expected = (
            ['h', 'r', 'd', 'r', 'h', 'a', 'a'],
            ['h', 't1', 't3', 't4', 'h', 'p', 'p'],
            ['h', 't6', 'p', 't7', 'h', 't5', 't3']
        )
        prev_line_expected = 'h c1 t1 t2 t3 t4 c3 c4 h c5 c6 c7 c8'
        updated_line_expected = 'h c1 t6 t2 t7 c3 c4 h c5 c6 t5 t3 c7 c8'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_single_hunk(self):
        prev_line    = 'c1 t1 t2 t3 t4 c3 c4'
        updated_line = 'c1 t6 t2 t7 c3 c4'

        diff_expected = (
            ['h', 'r', 'd', 'r'],
            ['h', 't1', 't3', 't4'],
            ['h', 't6', 'p', 't7']
        )
        prev_line_expected = 'h c1 t1 t2 t3 t4 c3 c4'
        updated_line_expected = 'h c1 t6 t2 t7 c3 c4'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_no_changes(self):
        prev_line    = 'c1 c2 c3 c4 c5 c6'
        updated_line = 'c1 c2 c3 c4 c5 c6'

        diff_expected = (
            ['h'],
            ['h'],
            ['h']
        )
        prev_line_expected = 'h c1 c2 c3 c4 c5 c6'
        updated_line_expected = 'h c1 c2 c3 c4 c5 c6'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_no_changes_in_long_context_in_tail(self):
        prev_line    = 'c1 t1 t2 t3 t4 c3 c4 c5 c6 c7 c8 c9 c10'
        updated_line = 'c1 t6 t2 t7 c3 c4 c5 c6 c7 c8 c9 c10'

        diff_expected = (
            ['h', 'r', 'd', 'r', 'h'],  # in long tail only one hunk token is added because no changed tokens in tail
            ['h', 't1', 't3', 't4', 'h'],
            ['h', 't6', 'p', 't7', 'h']
        )
        prev_line_expected = 'h c1 t1 t2 t3 t4 c3 c4 h c5 c6 c7 c8 c9 c10'
        updated_line_expected = 'h c1 t6 t2 t7 c3 c4 h c5 c6 c7 c8 c9 c10'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_no_changes_in_long_context_in_middle(self):
        prev_line    = 'c0 c1 t1 c3 c4 c5 c6 c7 c8 t3 c9'
        updated_line = 'c0 c1 t2 c3 c4 c5 c6 c7 c8 t4 c9'

        diff_expected = (
            ['h', 'r', 'h', 'r'],
            ['h', 't1', 'h', 't3'],
            ['h', 't2', 'h', 't4']
        )
        prev_line_expected = 'h c0 c1 t1 c3 c4 h c5 c6 c7 c8 t3 c9'
        updated_line_expected = 'h c0 c1 t2 c3 c4 h c5 c6 c7 c8 t4 c9'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_no_changes_in_long_context_in_beginning(self):
        prev_line    = 'c-1 c-2 c-3 c-4 c0 c1 t1 c3 c4 t3 c5 c6'
        updated_line = 'c-1 c-2 c-3 c-4 c0 c1 t2 c3 c4 t4 c5 c6'

        diff_expected = (
            ['h', 'r', 'h', 'r'],
            ['h', 't1', 'h', 't3'],
            ['h', 't2', 'h', 't4']
        )
        prev_line_expected = 'h c-1 c-2 c-3 c-4 c0 c1 t1 c3 c4 h t3 c5 c6'
        updated_line_expected = 'h c-1 c-2 c-3 c-4 c0 c1 t2 c3 c4 h t4 c5 c6'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_only_changes(self):
        prev_line    = 't1 t2 t3 t4'
        updated_line = 't5 t6 t7 t8'

        diff_expected = (
            ['h', 'r', 'r', 'r', 'r'],
            ['h', 't1', 't2', 't3', 't4'],
            ['h', 't5', 't6', 't7', 't8']
        )
        prev_line_expected = 'h t1 t2 t3 t4'
        updated_line_expected = 'h t5 t6 t7 t8'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_no_preceding_context(self):
        prev_line    = 't1 c1 c2 t2 c1 c2 t3 c1 c2 t4 c1 c2'
        updated_line = 't5 c1 c2 t6 c1 c2 t7 c1 c2 t8 c1 c2'

        diff_expected = (
            ['h', 'r', 'h', 'r', 'h', 'r', 'h', 'r'],
            ['h', 't1', 'h', 't2', 'h', 't3', 'h', 't4'],
            ['h', 't5', 'h', 't6', 'h', 't7', 'h', 't8']
        )
        prev_line_expected = 'h t1 c1 c2 h t2 c1 c2 h t3 c1 c2 h t4 c1 c2'
        updated_line_expected = 'h t5 c1 c2 h t6 c1 c2 h t7 c1 c2 h t8 c1 c2'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_context_only_between_changes(self):
        prev_line    = 't1 c1 c2 t2 c1 c2 t3 c1 c2 t4'
        updated_line = 't5 c1 c2 t6 c1 c2 t7 c1 c2 t8'

        diff_expected = (
            ['h', 'r', 'h', 'r', 'h', 'r', 'h', 'r'],
            ['h', 't1', 'h', 't2', 'h', 't3', 'h', 't4'],
            ['h', 't5', 'h', 't6', 'h', 't7', 'h', 't8']
        )
        prev_line_expected = 'h t1 c1 c2 h t2 c1 c2 h t3 c1 c2 h t4'
        updated_line_expected = 'h t5 c1 c2 h t6 c1 c2 h t7 c1 c2 h t8'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)

    def test_diff_sequences_and_add_hunks_context_no_succeeding_context(self):
        prev_line    = 'c1 c2 t2 c1 c2 t3 c1 c2 t4'
        updated_line = 'c1 c2 t6 c1 c2 t7 c1 c2 t8'

        diff_expected = (
            ['h', 'r', 'h', 'r', 'h', 'r'],
            ['h', 't2', 'h', 't3', 'h', 't4'],
            ['h', 't6', 'h', 't7', 'h', 't8']
        )
        prev_line_expected = 'h c1 c2 t2 c1 c2 h t3 c1 c2 h t4'
        updated_line_expected = 'h c1 c2 t6 c1 c2 h t7 c1 c2 h t8'

        diff, prev_line, updated_line = \
            HunkSplitterTest.HUNK_SPLITTER.diff_sequences_and_add_hunks(prev_line, updated_line)
        self.assertEqual(diff_expected, diff)
        self.assertEqual(prev_line_expected, prev_line)
        self.assertEqual(updated_line_expected, updated_line)


if __name__ == '__main__':
    unittest.main()

import unittest

from datasets.PatchNet.LevenshteinFilesPrevUpdatedGenerator import FilesDiffProcessor


class FilesDiffProcessorTest(unittest.TestCase):
    FILES_DIFF_PROCESSOR = FilesDiffProcessor(context_size=3)

    def test_cut_sequences_one_replacement_block_less_than_context(self):
        prev_tokens = ['a', 'b', 'c', 'd', 'e', 'f']
        updated_tokens = ['a', 'b', 'r', 'q', 'e', 'f']
        expected_cut_sequences = (['a', 'b', 'c', 'd', 'e', 'f'], ['a', 'b', 'r', 'q', 'e', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_replacement_block_full_context(self):
        prev_tokens = ['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f']
        updated_tokens = ['a', 'a', 'b', 'r', 'q', 'e', 'f', 'f']
        expected_cut_sequences = (['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f'], ['a', 'a', 'b', 'r', 'q', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_replacement_block_more_than_context(self):
        prev_tokens = ['extra-1', 'extra0', 'extra1', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'f', 'extra2', 'extra3']
        updated_tokens = ['extra-1', 'extra0', 'extra1', 'a', 'a', 'b', 'r', 'q', 'e', 'f', 'f', 'extra2', 'extra3']
        expected_cut_sequences = (['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f'],
                                  ['a', 'a', 'b', 'r', 'q', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_deletion_block_less_than_context_right(self):
        prev_tokens = ['a', 'a', 'b', 'c', 'd', 'e', 'f']
        updated_tokens = ['a', 'a', 'b', 'e', 'f']
        expected_cut_sequences = (['a', 'a', 'b', 'c', 'd', 'e', 'f'],
                                  ['a', 'a', 'b', 'e', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_deletion_block_full_context(self):
        prev_tokens = ['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f']
        updated_tokens = ['a', 'a', 'b', 'e', 'f', 'f']
        expected_cut_sequences = (['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f'],
                                  ['a', 'a', 'b', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_deletion_block_more_than_context(self):
        prev_tokens = ['extra-1', 'extra0', 'extra1', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'f', 'extra2', 'extra3']
        updated_tokens = ['extra-1', 'extra0', 'extra1', 'a', 'a', 'b', 'e', 'f', 'f', 'extra2', 'extra3']
        expected_cut_sequences = (['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f'],
                                  ['a', 'a', 'b', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_addition_block_less_than_context_left(self):
        prev_tokens = ['a', 'b', 'e', 'f', 'f']
        updated_tokens = ['a', 'b', 'r', 'q', 'e', 'f', 'f']
        expected_cut_sequences = (['a', 'b', 'e', 'f', 'f'], ['a', 'b', 'r', 'q', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_addition_block_full_context(self):
        prev_tokens = ['a', 'a', 'b', 'e', 'f', 'f']
        updated_tokens = ['a', 'a', 'b', 'r', 'q', 'e', 'f', 'f']
        expected_cut_sequences = (['a', 'a', 'b', 'e', 'f', 'f'],
                                  ['a', 'a', 'b', 'r', 'q', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_one_addition_block_more_than_context(self):
        prev_tokens = ['extra-1', 'extra0', 'extra1', 'a', 'a', 'b', 'e', 'f', 'f', 'extra2', 'extra3']
        updated_tokens = ['extra-1', 'extra0', 'extra1', 'a', 'a', 'b', 'r', 'q', 'e', 'f', 'f', 'extra2', 'extra3']
        expected_cut_sequences = (['a', 'a', 'b', 'e', 'f', 'f'],
                                  ['a', 'a', 'b', 'r', 'q', 'e', 'f', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_all_replaced(self):
        prev_tokens = ['a', 'b', 'c']
        updated_tokens = ['d', 'e', 'f']
        expected_cut_sequences = (['a', 'b', 'c'], ['d', 'e', 'f'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_all_deleted(self):
        prev_tokens = ['a', 'b', 'c']
        updated_tokens = []
        expected_cut_sequences = (['a', 'b', 'c'], [])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_all_added(self):
        prev_tokens = []
        updated_tokens = ['a', 'b', 'c']
        expected_cut_sequences = ([], ['a', 'b', 'c'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_padding_on_the_edges(self):
        prev_tokens = ['e', 'd', 'a', 'b', 'c']
        updated_tokens = ['a', 'b', 'c', 'f', 'g']
        expected_cut_sequences = (['e', 'd', 'a', 'b', 'c'], ['a', 'b', 'c', 'f', 'g'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_padding_on_the_edges_change_in_the_middle(self):
        prev_tokens = ['c1', 'c2', 'c3', 'c4', 'e', 'd', 'a', 'b', 'c', 'f', 'g', 'c1', 'c2', 'c3']
        updated_tokens = ['a', 'r', 'c']
        expected_cut_sequences = (['c1', 'c2', 'c3', 'c4', 'e', 'd', 'a', 'b', 'c', 'f', 'g', 'c1', 'c2', 'c3'],
                                  ['a', 'r', 'c'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_padding_three_separate_changes(self):
        prev_tokens = ['d1', 'd2', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'r1', 'r2', 'r3', 'c1', 'c2', 'c3',
                       'c4', 'c5', 'c6', 'c7', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
        updated_tokens = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'r5', 'r6', 'r7', 'c1', 'c2', 'c3',
                          'c4', 'c5', 'c6', 'c7', 'a1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
        expected_cut_sequences = (['d1', 'd2', 'c1', 'c2', 'c3', 'c5', 'c6', 'c7', 'r1', 'r2', 'r3', 'c1', 'c2',
                                   'c3', 'c5', 'c6', 'c7', 'c1', 'c2', 'c3'],
                                  ['c1', 'c2', 'c3', 'c5', 'c6', 'c7', 'r5', 'r6', 'r7', 'c1', 'c2', 'c3',
                                   'c5', 'c6', 'c7', 'a1', 'c1', 'c2', 'c3'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_padding_three_separate_changes_union(self):
        prev_tokens = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'd1', 'd2', 'c1', 'c2', 'r1', 'r2', 'r3', 'c1', 'c2', 'c3',
                       'c4', 'c5', 'c6', 'c7', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
        updated_tokens = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c1', 'c2', 'r5', 'r6', 'r7', 'c1', 'c2', 'c3',
                          'c4', 'c5', 'c6', 'c7', 'a1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
        expected_cut_sequences = (['c4', 'c5', 'c6', 'd1', 'd2', 'c1', 'c2', 'r1', 'r2', 'r3',
                                   'c1', 'c2', 'c3', 'c5', 'c6', 'c7', 'c1', 'c2', 'c3'],
                                  ['c4', 'c5', 'c6', 'c1', 'c2', 'r5', 'r6', 'r7', 'c1', 'c2', 'c3',
                                   'c5', 'c6', 'c7', 'a1', 'c1', 'c2', 'c3'])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)

    def test_cut_sequences_no_changes(self):
        prev_tokens = ['a', 'b', 'c']
        updated_tokens = ['a', 'b', 'c']
        expected_cut_sequences = ([], [])

        prev_tokens = [(None, t) for t in prev_tokens]
        updated_tokens = [(None, t) for t in updated_tokens]
        expected_cut_sequences = (
            [(None, t) for t in expected_cut_sequences[0]], [(None, t) for t in expected_cut_sequences[1]])

        cut_sequences = FilesDiffProcessorTest.FILES_DIFF_PROCESSOR.cut_sequences(prev_tokens, updated_tokens)
        self.assertEqual(expected_cut_sequences, cut_sequences)


if __name__ == '__main__':
    unittest.main()

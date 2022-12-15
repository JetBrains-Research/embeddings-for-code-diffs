import unittest

from datasets.PatchNet.PatchNetDataset import DataSample
from datasets.PatchNet.canonicalization import canonicalize_sample


class MyTestCase(unittest.TestCase):
    def test_canonicalization_tokens_to_leave_are_left(self):
        tokens_to_leave = {'type1': {'a'}, 'type2': {'b', 'd'}, 'type3': {'c'}}
        prev_tokens = [('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a')]
        updated_tokens = [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd')]
        expected_canonicalized = (prev_tokens, updated_tokens)
        canonicalized = canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens)
        self.assertEqual(expected_canonicalized, canonicalized)

    def test_canonicalization_same_special_token_in_both(self):
        tokens_to_leave = {'type1': {'a'}, 'type2': {}, 'type3': {'c'}}
        prev_tokens = [('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a')]
        updated_tokens = [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd')]
        expected_canonicalized = ([('type1', 'a'), ('type2', '<type2_0>'), ('type3', 'c'), ('type2', '<type2_1>'), ('type1', 'a')],
                                  [('type1', 'a'), ('type1', 'a'), ('type2', '<type2_0>'), ('type3', 'c'), ('type2', '<type2_1>')])
        canonicalized = canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens)
        self.assertEqual(expected_canonicalized, canonicalized)

    def test_canonicalization_special_token_in_prev(self):
        tokens_to_leave = {'type1': {'a'}, 'type2': {'b', 'd'}, 'type3': {'c'}}
        prev_tokens = [('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a'), ('type1', 'e')]
        updated_tokens = [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd')]
        expected_canonicalized = ([('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a'), ('type1', '<type1_0>')],
                                  [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd')])
        canonicalized = canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens)
        self.assertEqual(expected_canonicalized, canonicalized)

    def test_canonicalization_special_token_in_updated(self):
        tokens_to_leave = {'type1': {'a'}, 'type2': {'b', 'd'}, 'type3': {'c'}}
        prev_tokens = [('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a')]
        updated_tokens = [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type3', 'e')]
        expected_canonicalized = ([('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a')],
                                  [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type3', '<type3_0>')])
        canonicalized = canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens)
        self.assertEqual(expected_canonicalized, canonicalized)

    def test_canonicalization_special_token_in_prev_and_updated(self):
        tokens_to_leave = {'type1': {'a'}, 'type2': {'b', 'd'}, 'type3': {'c'}}
        prev_tokens = [('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a'), ('type3', 'e0')]
        updated_tokens = [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type3', 'e1')]
        expected_canonicalized = ([('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a'), ('type3', '<type3_0>')],
                                  [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type3', '<type3_1>')])
        canonicalized = canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens)
        self.assertEqual(expected_canonicalized, canonicalized)

    def test_canonicalization_special_token_in_prev_and_updated_different_types(self):
        tokens_to_leave = {'type1': {'a'}, 'type2': {'b', 'd'}, 'type3': {'c'}}
        prev_tokens = [('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a'), ('type1', 'e0')]
        updated_tokens = [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type3', 'e1')]
        expected_canonicalized = ([('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type1', 'a'), ('type1', '<type1_0>')],
                                  [('type1', 'a'), ('type1', 'a'), ('type2', 'b'), ('type3', 'c'), ('type2', 'd'), ('type3', '<type3_0>')])
        canonicalized = canonicalize_sample(tokens_to_leave, prev_tokens, updated_tokens)
        self.assertEqual(expected_canonicalized, canonicalized)


if __name__ == '__main__':
    unittest.main()

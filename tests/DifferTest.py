import unittest

from edit_representation.sequence_encoding.Differ import Differ

REPLACEMENT_TOKEN = 'замена'
DELETION_TOKEN = 'удаление'
ADDITION_TOKEN = 'добавление'
UNCHANGED_TOKEN = 'равенство'
PADDING_TOKEN = 'паддинг'


class DifferTest(unittest.TestCase):
    differ = Differ(REPLACEMENT_TOKEN, DELETION_TOKEN, ADDITION_TOKEN, UNCHANGED_TOKEN, PADDING_TOKEN)

    def check_diffs(self, diff):
        for i in range(len(diff[0])):
            if diff[0][i] == UNCHANGED_TOKEN:
                self.assertEqual(diff[1][i], diff[2][i])
            elif diff[0][i] == REPLACEMENT_TOKEN:
                self.assertNotEqual(diff[1][i], diff[2][i])
            elif diff[0][i] == ADDITION_TOKEN:
                self.assertEqual(diff[1][i], PADDING_TOKEN)
                self.assertNotEqual(diff[2][i], PADDING_TOKEN)
            elif diff[0][i] == DELETION_TOKEN:
                self.assertNotEqual(diff[1][i], PADDING_TOKEN)
                self.assertEqual(diff[2][i], PADDING_TOKEN)

    def test_no_empty_strings_in_output(self):
        prev_tokens = 'Assert . Equal ( LITERAL , VAR0 . Relational ( ) . ConstraintName ) ; var VAR1 = VAR0 . ' \
                      'DeclaringEntityType ; Assert . Equal ( nameof ( VAR2 . VAR3 ) , VAR1 . FindPrimaryKey ( ) . VAR4 [ 0 ' \
                      '] . VAR5 ) ; Assert . Equal ( LITERAL , VAR1 . GetKeys ( ) . VAR6 ( ) . Relational ( ) . VAR5 ) ; ' \
                      'Assert . Equal ( LITERAL , VAR1 . GetIndexes ( ) . Count ( ) ) ; var VAR7 = VAR1 . GetIndexes ( ) . ' \
                      'First ( ) ; Assert . Equal ( LITERAL , VAR7 . VAR4 [ 0 ] . VAR5 ) ; Assert . True ( VAR7 . IsUnique ) ' \
                      '; Assert . Equal ( LITERAL , VAR7 . Relational ( ) . Filter ) ; var VAR8 = VAR1 . GetIndexes ( ) . ' \
                      'Last ( ) ; Assert . Equal ( LITERAL , VAR8 . VAR4 [ 0 ] . VAR5 ) ; Assert . False ( VAR8 . IsUnique ) ' \
                      '; Assert . Null ( VAR8 . Relational ( ) . Filter ) ; Assert . Equal ( new object [ ] { 1 , - 1 } , ' \
                      'VAR1 . GetSeedData ( ) . VAR6 ( ) . Values ) ; Assert . Equal ( nameof ( VAR9 ) , VAR1 . Relational ( ' \
                      ') . TableName ) ;'.split()
        updated_tokens = 'Assert . Equal ( LITERAL , VAR0 . GetConstraintName ( ) ) ; var VAR1 = VAR0 . DeclaringEntityType ' \
                         '; Assert . Equal ( nameof ( VAR2 . VAR3 ) , VAR1 . FindPrimaryKey ( ) . VAR4 [ 0 ] . VAR5 ) ; ' \
                         'Assert . Equal ( LITERAL , VAR1 . GetKeys ( ) . VAR6 ( ) . GetName ( ) ) ; Assert . Equal ( ' \
                         'LITERAL , VAR1 . GetIndexes ( ) . Count ( ) ) ; var VAR7 = VAR1 . GetIndexes ( ) . First ( ) ; ' \
                         'Assert . Equal ( LITERAL , VAR7 . VAR4 [ 0 ] . VAR5 ) ; Assert . True ( VAR7 . IsUnique ) ; Assert ' \
                         '. Equal ( LITERAL , VAR7 . GetFilter ( ) ) ; var VAR8 = VAR1 . GetIndexes ( ) . Last ( ) ; Assert ' \
                         '. Equal ( LITERAL , VAR8 . VAR4 [ 0 ] . VAR5 ) ; Assert . False ( VAR8 . IsUnique ) ; Assert . ' \
                         'Null ( VAR8 . GetFilter ( ) ) ; Assert . Equal ( new object [ ] { 1 , - 1 } , VAR1 . GetSeedData ( ' \
                         ') . VAR6 ( ) . Values ) ; Assert . Equal ( nameof ( VAR9 ) , VAR1 . GetTableName ( ) ) ; '.split()
        diff = self.differ.diff_tokens_fast_lvn_all_aligned(prev_tokens, updated_tokens)
        self.assertFalse('' in diff[0])
        self.assertFalse('' in diff[1])
        self.assertFalse('' in diff[2])
        self.check_diffs(diff)

    def test_empty(self):
        prev_tokens = []
        updated_tokens = []
        diff_expected = ([], [], [])
        diff = self.differ.diff_tokens_fast_lvn_all_aligned(prev_tokens, updated_tokens)
        self.assertEqual(diff_expected, diff)

    def test_real_example_from_dataset(self):
        prev_tokens = 'public void METHOD_1 ( TYPE_1 VAR_1 ) { VAR_2 . add ( VAR_1 ) ; VAR_3 . METHOD_2 ( ) ; }'.split()
        updated_tokens = 'public void METHOD_1 ( TYPE_1 VAR_1 ) { }'.split()
        diff_expected = (
            [UNCHANGED_TOKEN, UNCHANGED_TOKEN, UNCHANGED_TOKEN, UNCHANGED_TOKEN,
             UNCHANGED_TOKEN, UNCHANGED_TOKEN, UNCHANGED_TOKEN, UNCHANGED_TOKEN,
             DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN,
             DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN,
             DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN, DELETION_TOKEN,
             UNCHANGED_TOKEN],
            prev_tokens,
            updated_tokens[:8] + [PADDING_TOKEN for _ in range(13)] + updated_tokens[8:]
        )
        diff = self.differ.diff_tokens_fast_lvn_all_aligned(prev_tokens, updated_tokens)
        self.assertEqual(diff_expected, diff)

    def test_from_article(self):
        prev_tokens = ['v', '.', 'F', '=', 'x', '+', 'x']
        updated_tokens = ['u', '=', 'x', '+', 'x', ';']
        diff_expected = (
            [DELETION_TOKEN, DELETION_TOKEN, REPLACEMENT_TOKEN, UNCHANGED_TOKEN,
             UNCHANGED_TOKEN, UNCHANGED_TOKEN, UNCHANGED_TOKEN, ADDITION_TOKEN],
            ['v', '.', 'F', '=', 'x', '+', 'x', PADDING_TOKEN],
            [PADDING_TOKEN, PADDING_TOKEN, 'u', '=', 'x', '+', 'x', ';']
        )
        diff = self.differ.diff_tokens_fast_lvn_all_aligned(prev_tokens, updated_tokens)
        self.assertEqual(diff_expected, diff)


if __name__ == '__main__':
    unittest.main()

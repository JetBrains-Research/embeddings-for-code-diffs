import unittest
from collections import Counter

from datasets.PatchNet.tokenizers import PygmentsCTokenizer


class PygmentsCTokenizerTest(unittest.TestCase):
    TOKENIZER = PygmentsCTokenizer()

    def test_multi_line_comments_are_banned(self):
        source_code = '/* Bluetooth HCI event handling. */'
        tokens_expected = ([], Counter())
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

        source_code = """    /* Set discovery state to stopped if we're not doing LE active
    * scanning.
    */"""
        tokens_expected = ([], Counter())
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

    def test_single_line_comments_are_banned(self):
        source_code = '// Bluetooth HCI event handling.'
        tokens_expected = ([], Counter())
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

    def test_line_separators_are_banned(self):
        source_code = """
        struct hci_conn *conn;
	    void *sent;
	    """
        tokens_expected = (['struct', 'hci_conn', '*', 'conn', ';', 'void', '*', 'sent', ';'],
                           Counter({'hci_conn': 1, 'conn': 1, 'sent': 1}))
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

    def test_string_literals_are_banned(self):
        source_code = """BT_DBG("%s status 0x%2.2x", hdev->name, status);"""
        tokens_expected = (['BT_DBG', '(', ',', 'hdev', '-', '>', 'name', ',', 'status', ')', ';'],
                           Counter({'BT_DBG': 1, 'hdev': 1, 'name': 1, 'status': 1}))
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

    def test_error_codes_are_not_banned(self):
        source_code = """if (!conn->out)
        return 0;"""
        tokens_expected = (['if', '(', '!', 'conn', '-', '>', 'out', ')', 'return', '0', ';'],
                           Counter({'conn': 1, 'out': 1}))
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

        source_code = """return 1;"""
        tokens_expected = (['return', '1', ';'], Counter())
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

    def test_numbers_are_banned(self):
        source_code = """if (pin_len == 16)
		conn->pending_sec_level = BT_SECURITY_HIGH;"""
        tokens_expected = \
            (['if', '(', 'pin_len', '=', '=', ')', 'conn', '-', '>', 'pending_sec_level', '=', 'BT_SECURITY_HIGH', ';'],
             Counter({'pin_len': 1, 'conn': 1, 'pending_sec_level': 1, 'BT_SECURITY_HIGH': 1}))
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)

    def test_identifier_names_counted_correctly(self):
        source_code = """hci_dev_lock(hdev);

	conn = hci_conn_hash_lookup_handle(hdev, __le16_to_cpu(rp->handle));
	if (conn)
		conn->role = rp->role;

	hci_dev_unlock(hdev);"""
        tokens_expected = \
            (['hci_dev_lock', '(', 'hdev', ')', ';', 'conn', '=', 'hci_conn_hash_lookup_handle', '(',
              'hdev', ',', '__le16_to_cpu', '(', 'rp', '-', '>', 'handle', ')', ')', ';',
              'if', '(', 'conn', ')', 'conn', '-', '>', 'role', '=', 'rp', '-', '>', 'role', ';',
              'hci_dev_unlock', '(', 'hdev', ')', ';'],
             Counter({'hci_dev_lock': 1, 'hdev': 3, 'conn': 3, 'hci_conn_hash_lookup_handle': 1,
                      '__le16_to_cpu': 1, 'rp': 2, 'handle': 1, 'role': 2, 'hci_dev_unlock': 1}))
        tokens_actual = self.TOKENIZER.tokenize(source_code)
        self.assertEqual(tokens_actual, tokens_expected)


if __name__ == '__main__':
    unittest.main()

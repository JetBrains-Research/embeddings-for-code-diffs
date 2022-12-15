from datasets.Jiang.utils import tokenize_git_diff_output_string


class DatasetFilter:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def validate(git_diff_output: str, msg: str) -> bool:
        tokens_per_line = tokenize_git_diff_output_string(git_diff_output)
        counters = {'mmm': 0, 'ppp': 0,
                    'deleted file mode': 0, 'new file mode': 0,
                    'rename from': 0, 'rename to': 0,
                    'old mode': 0, 'new mode': 0}
        max_prefix_len = 3
        for tokens_in_line in tokens_per_line:
            for i in range(1, max_prefix_len + 1):
                prefix_to_check = ' '.join(tokens_in_line[:i])
                if prefix_to_check in counters:
                    counters[prefix_to_check] += 1
        return counters['mmm'] == 1 and counters['ppp'] == 1 and \
            counters['deleted file mode'] == 0 and counters['new file mode'] == 0 and \
            counters['rename from'] == 0 and counters['rename to'] == 0

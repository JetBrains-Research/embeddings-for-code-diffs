import subprocess
from pathlib import Path

from neural_editor.seq2seq.config import Config


class BleuCalculation:
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def conduct(self, path_to_predicted: str, path_to_target: str, dataset_label: str) -> None:
        print(f'Start conducting BLEU calculation experiment for {dataset_label}...')
        process = subprocess.Popen([self.config['BLEU_PERL_SCRIPT_PATH'], path_to_target], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = process.communicate(input=Path(path_to_predicted).read_bytes())
        print(result[0])
        print(f'Errors: {result[1]}')

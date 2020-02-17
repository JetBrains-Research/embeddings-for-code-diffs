import subprocess
import tempfile
from typing import List

from torchtext.data import Dataset

from neural_editor.seq2seq.config import Config


class BleuCalculation:
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def conduct(self, predictions: List[List[List[str]]], dataset: Dataset, dataset_label: str) -> None:
        top_1_predictions = ['' if len(prediction) == 0 else ' '.join(prediction[0]) for prediction in predictions]
        targets = [' '.join(example.trg) for example in dataset]
        with tempfile.NamedTemporaryFile(mode='w') as file_with_targets:
            file_with_targets.write('\n'.join(targets))
            file_with_targets.flush()
            print(f'Start conducting BLEU calculation experiment for {dataset_label}...')
            process = subprocess.Popen([self.config['BLEU_PERL_SCRIPT_PATH'], file_with_targets.name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = process.communicate(input=('\n'.join(top_1_predictions)).encode())
            print(result[0])
            print(f'Errors: {result[1]}')

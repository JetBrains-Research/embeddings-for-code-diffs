"""
This file runs both training and evaluation
"""
import sys
from pathlib import Path

from datasets.CodeChangesDataset import CodeChangesTokensDataset
from datasets.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from neural_editor.seq2seq.analyze import test_neural_editor_model, test_commit_message_generation_model
from neural_editor.seq2seq.config import load_config
from neural_editor.seq2seq.train import run_train


def main():
    is_test = len(sys.argv) > 1 and sys.argv[1] == 'test'
    config_path = None if len(sys.argv) < 3 else Path(sys.argv[2])
    config = load_config(is_test, config_path)
    print('\n====STARTING TRAINING====\n', end='')
    train_dataset, val_dataset, test_dataset, diffs_field = \
        CodeChangesTokensDataset.load_data(config['VERBOSE'], config)
    fields = (diffs_field, diffs_field, diffs_field)
    neural_editor = run_train(train_dataset, val_dataset, fields,
                              'neural_editor', edit_encoder=None, config=config)
    train_dataset_commit, val_dataset_commit, test_dataset_commit, fields_commit = \
        CommitMessageGenerationDataset.load_data(diffs_field, config['VERBOSE'], config)
    commit_message_generator = run_train(train_dataset_commit, val_dataset_commit, fields_commit,
                                         'commit_msg_generator', neural_editor.edit_encoder, config=config)
    print('\n====STARTING NEURAL EDITOR EVALUATION====\n', end='')
    test_neural_editor_model(neural_editor, config)
    print('\n====STARTING COMMIT MSG GENERATOR EVALUATIONN====\n', end='')
    test_commit_message_generation_model(commit_message_generator, config)


if __name__ == "__main__":
    main()

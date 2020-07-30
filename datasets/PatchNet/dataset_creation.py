import pickle
import random
import sys
from collections import Counter, defaultdict
from datetime import timezone
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

from pydriller import RepositoryMining, ModificationType
from tqdm.auto import tqdm
from datasets.PatchNet.PatchNetDataset import PatchNetDataset
from datasets.PatchNet.tokenizers import PygmentsCTokenizer
from datasets.dataset_utils import get_indices_for_train_val_test
from neural_editor.seq2seq.experiments.PredictorMetricsCalculation import calculate_metrics
from neural_editor.seq2seq.test_utils import save_patchnet_metric_plot
import os

def get_changed_files(commit_hashes, linux_path) -> Tuple[List[int], List[int]]:
    c_extensions = ['.c', '.cpp', '.h', '.hpp', '.cc']
    changed_files_num = []
    changed_c_files_num = []
    for commit_hash in tqdm(commit_hashes):
        commits = list(RepositoryMining(str(linux_path.absolute()), single=commit_hash).traverse_commits())
        commit = commits[0]
        changed_files = 0
        changed_c_files = 0
        for modification in commit.modifications:
            changed_files += 1
            filename = modification.filename
            for c_extension in c_extensions:
                if filename.endswith(c_extension):
                    changed_c_files += 1
                    break
        changed_files_num.append(changed_files)
        changed_c_files_num.append(changed_c_files)
    return changed_files_num, changed_c_files_num


def extract_changed_files_num():
    if len(sys.argv) != 4:
        print('Usage: <root where to save processed data> <hash commits file> <path to linux repo>')
        exit(1)
    root = Path(sys.argv[1])
    linux_path = Path(sys.argv[3])
    commit_hashes = Path(sys.argv[2]).read_text().splitlines(keepends=False)[::2]
    commit_hashes = [commit_hash.split(': ')[-1] for commit_hash in commit_hashes]
    changed_files, changed_c_files = get_changed_files(commit_hashes, linux_path)
    root.joinpath('changed_files.txt').write_text('\n'.join([str(t) for t in changed_files]))
    root.joinpath('changed_c_files.txt').write_text('\n'.join([str(t) for t in changed_c_files]))


def print_changed_files_information():
    if len(sys.argv) != 2:
        print('Usage: <file with changed files numbers>')
        exit(1)
    changed_files_numbers = np.array([int(l) for l in Path(sys.argv[1]).read_text().splitlines(keepends=False)])
    print(f'Mean: {np.mean(changed_files_numbers)}')
    print(f'Std: {np.std(changed_files_numbers)}')
    print(f'Median: {np.median(changed_files_numbers)}')
    print(f'Min: {np.min(changed_files_numbers)}')
    print(f'Max: {np.max(changed_files_numbers)}')


def extract_metric_from_line(line, metric_type) -> float:
    return float(line.split(metric_type)[-1].split()[0].split(',')[0])


def draw_metrics_plots_patchnet_training():
    if len(sys.argv) != 4:
        print('Usage: <root where to save processed data> <path to training log file> <frequency_type>')
        exit(1)
    root = Path(sys.argv[1])
    training_log_lines = Path(sys.argv[2]).read_text().splitlines(keepends=False)
    frequency_type = sys.argv[3]
    metrics = read_metrics(frequency_type, training_log_lines)
    for metric_type in metrics:
        save_patchnet_metric_plot(metrics[metric_type], metric_type, str(root.absolute()))


def read_metrics(frequency_type, training_log_lines) -> Dict[str, List[float]]:
    metrics = {'loss': [], 'acc': []}
    if frequency_type == 'steps':
        for line in training_log_lines:
            for metric_type in metrics:
                if metric_type in line:
                    metric_value = extract_metric_from_line(line, metric_type)
                    metrics[metric_type].append(metric_value)
    elif frequency_type == 'epochs':
        epoch_metrics = {'loss': [], 'acc': []}
        for line in training_log_lines:
            for metric_type in metrics:
                if metric_type + ' ' in line:
                    metric_value = extract_metric_from_line(line, metric_type)
                    epoch_metrics[metric_type].append(metric_value)
            if 'at epoch' in line:
                for metric_type in epoch_metrics:
                    metrics[metric_type].append(np.mean(epoch_metrics[metric_type]))
                epoch_metrics = {'loss': [], 'acc': []}
    else:
        raise Exception('Unknown frequency type! Possible values: epochs, steps.')
    return metrics


def get_timestamps(commit_hashes, linux_path) -> List[float]:
    timestamps = []
    for commit_hash in tqdm(commit_hashes):
        commits = list(RepositoryMining(str(linux_path.absolute()), single=commit_hash).traverse_commits())
        commit = commits[0]
        timestamp = commit.author_date.replace(tzinfo=timezone.utc).timestamp()
        timestamps.append(timestamp)
    return timestamps


def extract_timestamps():
    if len(sys.argv) != 4:
        print('Usage: <root where to save processed data> <hash commits file> <path to linux repo>')
        exit(1)
    root = Path(sys.argv[1])
    linux_path = Path(sys.argv[3])
    commit_hashes = Path(sys.argv[2]).read_text().splitlines(keepends=False)[::2]
    commit_hashes = [commit_hash.split(': ')[-1] for commit_hash in commit_hashes]
    commit_timestamps = get_timestamps(commit_hashes, linux_path)
    root.joinpath('timestamps.txt').write_text('\n'.join([str(t) for t in commit_timestamps]))


def create_k_folds():
    if len(sys.argv) != 4:
        print('Usage: <root where to save processed data> <timestamps file> <number of folds>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']
    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    timestamps = [float(l) if l != 'None' else None for l in Path(sys.argv[2]).read_text().splitlines(keepends=False)]
    timestamps = [timestamps[int(data_sample[3])] for data_sample in data]
    k = int(sys.argv[3])
    folds = split_on_folds(data, k, timestamps)
    double_folds = folds + folds
    test_id = k - 1
    for i in range(k):
        data_to_write = {
            'test': double_folds[test_id],
            'val': double_folds[test_id - 1],
            'train': [el for l in double_folds[test_id - 4:test_id - 1] for el in l]
        }
        fold_folder = root.joinpath(f'fold_{i + 1}')
        fold_folder.mkdir()
        for k, v in data_to_write.items():
            folder = fold_folder.joinpath(k)
            folder.mkdir()
            filenames_lines = {filename: [] for filename in filenames}
            for v_data_sample in v:
                for i, filename in enumerate(filenames_lines):
                    filenames_lines[filename].append(v_data_sample[i])
            for filename, lines in filenames_lines.items():
                folder.joinpath(filename).write_text('\n'.join(lines))
        test_id += 1


def create_k_folds_for_patchnet():
    if len(sys.argv) != 6:
        print('Usage: <path to out file> <commit hashes file> <timestamps file> <number of folds> <path where to save data>')
        exit(1)
    patchnet_out_file = Path(sys.argv[1])
    data = read_patchnet_out_file(patchnet_out_file)
    commit_hashes = [l.split(':')[0] for l in Path(sys.argv[2]).read_text().splitlines(keepends=False)]
    commit_hashes_to_id = {h: i for i, h in enumerate(commit_hashes)}
    timestamps = [float(l) for l in Path(sys.argv[3]).read_text().splitlines(keepends=False)]
    timestamps = [timestamps[commit_hashes_to_id[sample[1]]] for sample in data]
    k = int(sys.argv[4])
    root = Path(sys.argv[5])
    folds = split_on_folds(data, k, timestamps)
    double_folds = folds + folds
    test_id = k - 1
    for i in range(k):
        data_to_write = {
            'test': double_folds[test_id],
            'training': [el for l in double_folds[test_id - 4:test_id] for el in l]
        }
        fold_folder = root.joinpath(f'fold_{i + 1}')
        fold_folder.mkdir(exist_ok=True)
        for k, v in data_to_write.items():
            file_to_write = fold_folder.joinpath(f'{k}_data.out')
            lines_to_write = [l for sample in v for l in sample[0]]
            file_to_write.write_text('\n'.join(lines_to_write))
        test_id += 1


def split_on_folds(data, k, timestamps):
    sort_idx = np.argsort(timestamps)
    sorted_data = []
    for idx in sort_idx:
        sorted_data.append(data[idx])
    folds = []
    fold_size = round(len(sorted_data) / k)
    cur_idx = 0
    for i in range(k):
        next_idx = len(sorted_data) if i + 1 == k else cur_idx + fold_size
        folds.append(sorted_data[cur_idx: next_idx])
        cur_idx = next_idx
    return folds


def remove_examples_by_hash_from_patchnet(data_filepath: Path, hashes_to_remove):
    data = read_patchnet_out_file(data_filepath)
    filtered_data = [sample for sample in data if sample[1] not in hashes_to_remove]
    lines_to_write = [l for sample in filtered_data for l in sample[0]]
    data_filepath.write_text('\n'.join(lines_to_write))


def read_patchnet_out_file(data_filepath):
    lines = [l for l in data_filepath.read_text().splitlines(keepends=False)]
    data_sample_start_id = [i for i, l in enumerate(lines) if l.startswith('commit: ')]
    data = []
    for i in range(len(data_sample_start_id)):
        commit_hash = lines[data_sample_start_id[i]].split(': ')[-1]
        if i != len(data_sample_start_id) - 1:
            data.append((lines[data_sample_start_id[i]:data_sample_start_id[i + 1]], commit_hash))
        else:
            data.append((lines[data_sample_start_id[i]:], commit_hash))
    return data


def remove_from_dataset_by_ids(root: Path, ids_to_remove):
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']

    old_data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    data = [sample for sample in old_data if int(sample[3]) not in ids_to_remove]
    filenames_lines = {filename: [] for filename in filenames}
    for data_sample in data:
        for i, filename in enumerate(filenames_lines):
            filenames_lines[filename].append(data_sample[i])
    for filename, lines in filenames_lines.items():
        root.joinpath(filename).write_text('\n'.join(lines))


def keep_only_intersection_of_commits():
    if len(sys.argv) != 4:
        print('Usage: <patchnet out file> <commit hashes file> <dataset root file>')
        exit(1)
    patchnet_hashes = set([l.split(': ')[-1] for l in Path(sys.argv[1]).read_text().splitlines(keepends=False) if 'commit: ' in l])
    commit_hashes = [l.split(':')[0] for l in Path(sys.argv[2]).read_text().splitlines(keepends=False)]
    commit_hashes_to_id = {h: i for i, h in enumerate(commit_hashes)}
    mine_dataset_root = Path(sys.argv[3])
    mine_commit_hashes = set([commit_hashes[int(l)] for l in mine_dataset_root.joinpath('ids.txt').read_text().splitlines(keepends=False)])
    intersection = patchnet_hashes & mine_commit_hashes
    only_patchnet = patchnet_hashes - mine_commit_hashes
    only_mine = mine_commit_hashes - patchnet_hashes
    only_mine_ids = set(commit_hashes_to_id[h] for h in only_mine)
    print(f'Intersection size: {len(intersection)}')
    print(f'Only patchnet: {len(only_patchnet)}')
    print(f'{only_patchnet}')
    print(f'Only mine: {len(only_mine)}')
    print(f'{only_mine}')
    to_be_left = len(intersection)
    print(f'To be left: {to_be_left} / {len(commit_hashes)} = {to_be_left / len(commit_hashes)}')
    remove_examples_by_hash_from_patchnet(Path(sys.argv[1]), only_patchnet)
    remove_from_dataset_by_ids(mine_dataset_root, only_mine_ids)


def split_on_train_test_val():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']
    folder_names = ['train', 'val', 'test']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    train_indices, val_indices, test_indices = get_indices_for_train_val_test(len(data), ratios=(0.1, 0.1))
    indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}
    print(f'Train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')
    for folder_name in folder_names:
        path_to_write = root.joinpath(folder_name)
        path_to_write.mkdir(exist_ok=True)
        folder_indices = indices[folder_name]
        filenames_lines = {filename: [] for filename in filenames}
        for idx in folder_indices:
            data_sample = data[idx]
            for i, filename in enumerate(filenames_lines):
                filenames_lines[filename].append(data_sample[i])
        for filename, lines in filenames_lines.items():
            path_to_write.joinpath(filename).write_text('\n'.join(lines))


def cut_dataset(n, shuffle=False):
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    if shuffle:
        random.shuffle(data)
    data = data[:n]
    filenames_lines = {filename: [] for filename in filenames}
    for data_sample in data:
        for i, filename in enumerate(filenames_lines):
            filenames_lines[filename].append(data_sample[i])
    for filename, lines in filenames_lines.items():
        root.joinpath(filename).write_text('\n'.join(lines))


def partition_data():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt', 'trg.txt', 'ids.txt']
    folder_names = ['neural_editor', 'predictor']

    data = list(zip(*[root.joinpath(filename).read_text().splitlines(keepends=False) for filename in filenames]))
    ne_indices, predictor_indices, _ = get_indices_for_train_val_test(len(data), ratios=(0.5, 0))
    indices = {'neural_editor': ne_indices, 'predictor': predictor_indices}
    print(f'Neural editor: {len(ne_indices)}, predictor: {len(predictor_indices)}')
    for folder_name in folder_names:
        path_to_write = root.joinpath(folder_name)
        path_to_write.mkdir(exist_ok=True)
        folder_indices = indices[folder_name]
        filenames_lines = {filename: [] for filename in filenames}
        for idx in folder_indices:
            data_sample = data[idx]
            for i, filename in enumerate(filenames_lines):
                filenames_lines[filename].append(data_sample[i])
        for filename, lines in filenames_lines.items():
            path_to_write.joinpath(filename).write_text('\n'.join(lines))


def mine_dataset() -> None:
    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print('Usage: <root where to save processed data> <path to local copy of linux git repository> '
              '<path to file with description of dataset>')
        exit(1)
    root = Path(sys.argv[1])
    linux_repository_filepath = Path(sys.argv[2])
    dataset_description_file = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    if not root.is_dir():
        print(f'No such directory: {root.absolute()}')
    if dataset_description_file is not None and not dataset_description_file.is_file():
        print(f'No such file: {dataset_description_file.absolute()}')
        exit(1)
    if not linux_repository_filepath.is_dir():
        print(f'No such directory: {linux_repository_filepath.absolute()}')
        exit(1)
    patch_net_dataset = PatchNetDataset(root, dataset_description_file, linux_repository_filepath)
    patch_net_dataset.print_statistics()
    patch_net_dataset.write_data()


def apply_tokenizer_again():
    if len(sys.argv) != 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    filenames = ['prev.txt', 'updated.txt']
    counter = Counter()
    tokenizer = PygmentsCTokenizer()
    for filename in filenames:
        lines = root.joinpath(filename).read_text().splitlines(keepends=False)
        lines_to_save = []
        for line in tqdm(lines):
            tokens, line_counter = tokenizer.tokenize(line)
            lines_to_save.append(' '.join(tokens))
            counter += line_counter
        root.joinpath('filtered_' + filename).write_text('\n'.join(lines_to_save))
    with root.joinpath('identifier_names_counter.pkl').open('wb') as counter_file:
        pickle.dump(counter, counter_file)


def load_dataset() -> None:
    if len(sys.argv) < 2:
        print('Usage: <root where to save processed data>')
        exit(1)
    root = Path(sys.argv[1])
    dataset_description_file = None
    linux_repository_filepath = None
    if not root.is_dir():
        print(f'No such directory: {root.absolute()}')
    patch_net_dataset = PatchNetDataset(root, dataset_description_file, linux_repository_filepath)
    patch_net_dataset.load()
    print(patch_net_dataset.tokens_counter)
    print(patch_net_dataset.data_samples)


def convert_to_patchnet_format_list_of_commits():
    if len(sys.argv) < 4:
        print('Usage: <root> <commits_filename> <commits_new_filename>')
        exit(1)
    root = Path(sys.argv[1])
    commits_file = root.joinpath(sys.argv[2])
    commits_new_file = root.joinpath(sys.argv[3])
    commits_file_lines = commits_file.read_text().splitlines(keepends=False)
    commit_hashes = [l.split(': ')[-1] for l in commits_file_lines[::2]]
    commit_labels = [l.split(': ')[-1] for l in commits_file_lines[1::2]]
    new_lines = [f'{l[0]}: {l[1]}' for l in zip(commit_hashes, commit_labels)]
    commits_new_file.write_text('\n'.join(new_lines))


def calculate_performance_metrics_for_patchnet_model():
    # skip = [260, 282, 1133, 2494, 2677, 4704, 5214, 5693, 6562, 6926, 7533, 7998, 9544, 9682, 9816, 9897, 9998, 10325, 11029, 11226, 12268, 12748, 12837, 13349, 13432, 13608, 13742, 14146, 14585, 15609, 16185]
    skip = []

    if len(sys.argv) != 3:
        print('Usage: <test data out file> <root with data>')
        exit(1)
    test_labels = np.array([1 if l.split(': ')[-1] == 'true' else 0
                            for l in Path(sys.argv[1]).read_text().splitlines(keepends=False)
                            if l.startswith('label:')])
    test_labels = np.delete(test_labels, skip)
    root = Path(sys.argv[2])
    pred_probs = np.array([float(l) for l in root.joinpath('prediction.txt').read_text().splitlines(keepends=False)])
    pred_labels = pred_probs.round()
    metrics = calculate_metrics(test_labels, pred_labels, pred_probs)
    for metric, value in metrics.items():
        print(f'{metric}: {round(value, 3)}')
    root.joinpath('metrics.txt').write_text(str(metrics))


def concat_pre_train_datasets():
    if len(sys.argv) != 2:
        print('Usage: <root with data>')
        exit(1)
    root = Path(sys.argv[1])
    root_concat = root.joinpath('pre_train')
    root_concat.mkdir(exist_ok=True)
    concatenated_dataset = PatchNetDataset(root_concat, None, None)
    for folder in os.listdir(str(root)):
        if folder.startswith('pre_train_'):
            pre_train_root = root.joinpath(folder)
            pre_train_dataset = PatchNetDataset(pre_train_root, None, None)
            pre_train_dataset.load()
            concatenated_dataset.add(pre_train_dataset)
    concatenated_dataset.filter_empty()
    concatenated_dataset.write_data()


def unite_two_datasets():
    if len(sys.argv) != 4:
        print('Usage: <root with data 1> <root with data 2> <root for output>')
        exit(1)
    root_dataset1 = Path(sys.argv[1])
    root_dataset2 = Path(sys.argv[2])
    root_output = Path(sys.argv[3])

    dataset1 = PatchNetDataset(root_dataset1, None, None)
    dataset1.load()
    dataset2 = PatchNetDataset(root_dataset2, None, None)
    dataset2.load()
    output_dataset = PatchNetDataset(root_output, None, None)
    output_dataset.add(dataset1)
    output_dataset.add(dataset2)
    output_dataset.filter_empty()
    output_dataset.write_data()


if __name__ == "__main__":
    # cut_dataset(200, shuffle=False)
    # split_on_train_test_val()
    # partition_data()
    # create_k_folds()
    # create_k_folds_for_patchnet()
    # keep_only_intersection_of_commits()
    # convert_to_patchnet_format_list_of_commits()
    # extract_timestamps()
    mine_dataset()
    # load_dataset()
    # apply_tokenizer_again()
    # calculate_performance_metrics_for_patchnet_model()
    # draw_metrics_plots_patchnet_training()
    # extract_changed_files_num()
    # print_changed_files_information()
    # concat_pre_train_datasets()
    # unite_two_datasets()

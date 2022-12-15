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
    none_timestamps = [timestamps[int(data_sample[3])] for data_sample in data]
    none_data = data
    data = []
    timestamps = []
    print(f'Before {len(none_timestamps)}')
    for timestamp, data_sample in zip(none_timestamps, none_data):
        if timestamp is not None:
            data.append(data_sample)
            timestamps.append(timestamp)
    print(f'Before {len(timestamps)}')
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
    skip = [4, 5, 8, 26, 33, 53, 59, 61, 62, 66, 84, 89, 107, 116, 122, 127, 132, 138, 175, 193, 194, 204, 205, 206, 217, 223, 229, 243, 245, 276, 293, 312, 319, 327, 328, 349, 363, 372, 399, 406, 408, 428, 433, 441, 456, 458, 499, 515, 521, 524, 528, 536, 542, 544, 572, 596, 604, 613, 627, 638, 657, 658, 659, 679, 697, 701, 739, 749, 753, 766, 767, 779, 785, 792, 812, 820, 821, 824, 826, 849, 855, 865, 873, 894, 902, 940, 942, 948, 985, 1002, 1005, 1018, 1019, 1031, 1062, 1084, 1090, 1091, 1096, 1104, 1105, 1130, 1133, 1152, 1155, 1180, 1182, 1198, 1204, 1217, 1222, 1242, 1247, 1284, 1290, 1293, 1296, 1301, 1328, 1334, 1356, 1368, 1381, 1389, 1402, 1414, 1417, 1423, 1428, 1432, 1439, 1440, 1458, 1460, 1468, 1469, 1470, 1472, 1508, 1518, 1524, 1525, 1531, 1536, 1538, 1542, 1568, 1570, 1573, 1622, 1628, 1661, 1670, 1679, 1688, 1715, 1753, 1760, 1761, 1763, 1777, 1781, 1807, 1817, 1825, 1877, 1881, 1893, 1896, 1901, 1915, 1919, 1946, 1952, 1957, 1958, 1967, 1974, 1977, 2001, 2010, 2019, 2033, 2038, 2039, 2042, 2046, 2064, 2067, 2095, 2106, 2107, 2111, 2114, 2117, 2119, 2123, 2124, 2125, 2126, 2127, 2128, 2168, 2179, 2180, 2186, 2190, 2214, 2221, 2225, 2227, 2233, 2261, 2270, 2276, 2278, 2284, 2295, 2296, 2297, 2304, 2320, 2325, 2326, 2331, 2334, 2338, 2365, 2366, 2367, 2387, 2391, 2395, 2400, 2403, 2416, 2428, 2429, 2438, 2441, 2442, 2466, 2471, 2477, 2478, 2498, 2511, 2535, 2536, 2538, 2544, 2546, 2562, 2576, 2578, 2584, 2595, 2599, 2600, 2617, 2627, 2648, 2653, 2663, 2664, 2668, 2678, 2680, 2687, 2688, 2689, 2700, 2727, 2740, 2750, 2753, 2763, 2768, 2772, 2779, 2786, 2791, 2798, 2836, 2838, 2842, 2843, 2882, 2887, 2903, 2915, 2923, 2936, 2940, 2957, 2959, 2969, 2972, 2980, 2982, 2991, 2995, 2998, 3006, 3019, 3041, 3044, 3047, 3054, 3055, 3095, 3099, 3101, 3105, 3115, 3116, 3127, 3134, 3135, 3142, 3145, 3148, 3162, 3163, 3164, 3167, 3168, 3190, 3193, 3199, 3207, 3210, 3231, 3237, 3242, 3248, 3253, 3276, 3282, 3300, 3305, 3349, 3371, 3384, 3393, 3409, 3414, 3443, 3453, 3469, 3496, 3506, 3509, 3512, 3527, 3533, 3557, 3562, 3565, 3572, 3598, 3607, 3608, 3609, 3618, 3622, 3623, 3654, 3664, 3666, 3671, 3679, 3699, 3719, 3720, 3731, 3758, 3811, 3813, 3844, 3856, 3857, 3862, 3865, 3871, 3875, 3881, 3897, 3909, 3915, 3919, 3928, 3929, 3934, 3940, 3969, 3978, 3979, 4008, 4011, 4019, 4021, 4023, 4025, 4028, 4031, 4035, 4065, 4066, 4067, 4069, 4100, 4111, 4112, 4123, 4133, 4157, 4158, 4179, 4205, 4207, 4227, 4234, 4236, 4256, 4261, 4263, 4276, 4308, 4316, 4336, 4349, 4369, 4370, 4371, 4372, 4387, 4404, 4419, 4422, 4447, 4449, 4452, 4466, 4468, 4474, 4480, 4490, 4541, 4550, 4572, 4591, 4598, 4600, 4606, 4607, 4617, 4628, 4639, 4646, 4651, 4652, 4657, 4658, 4670, 4680, 4692, 4696, 4736, 4747, 4757, 4764, 4765, 4774, 4780, 4790, 4794, 4807, 4809, 4814, 4844, 4851, 4855, 4857, 4858, 4873, 4893, 4898, 4927, 4934, 4943, 4945, 4953, 4975, 4981, 4993, 4994, 5025, 5052, 5058, 5059, 5074, 5076, 5077, 5087, 5090, 5105, 5123, 5134, 5143, 5151, 5152, 5168, 5181, 5186, 5191, 5202, 5221, 5224, 5233, 5245, 5270, 5276, 5279, 5280, 5283, 5289, 5304, 5308, 5313, 5317, 5327, 5328, 5359, 5361, 5399, 5401, 5417, 5448, 5450, 5456, 5464, 5474, 5478, 5489, 5500, 5508, 5516, 5526, 5541, 5548, 5549, 5560, 5577, 5614, 5617, 5628, 5640, 5646, 5648, 5662, 5682, 5699, 5722, 5725, 5739, 5743, 5752, 5756, 5772, 5790, 5795, 5804, 5818, 5826, 5827, 5829, 5839, 5842, 5846, 5851, 5864, 5868, 5870, 5877, 5910, 5923, 5936, 5976, 6011, 6014, 6015, 6016, 6019, 6023, 6025, 6031, 6032, 6066, 6071, 6080, 6084, 6087, 6096, 6099, 6114, 6117, 6119, 6138, 6172, 6176, 6200, 6213, 6222, 6240, 6253, 6261, 6264, 6277, 6285, 6286, 6295, 6301, 6302, 6306, 6308, 6316, 6317, 6322, 6329, 6330, 6331, 6339, 6412, 6413, 6418, 6430, 6447, 6457, 6496, 6508, 6535, 6546, 6549, 6550, 6556, 6558, 6575, 6576, 6579, 6588, 6593, 6596, 6619, 6626, 6634, 6639, 6642, 6649, 6661, 6670, 6671, 6673, 6674, 6678, 6694, 6695, 6696, 6700, 6715, 6724, 6728, 6739, 6745, 6766, 6770, 6795, 6812, 6831, 6832, 6842, 6850, 6872, 6882, 6894, 6903, 6905, 6911, 6916, 6931, 6934, 6935, 6958, 6959, 6968, 6971, 6973, 6974, 6975, 7000, 7006, 7064, 7079, 7116, 7117, 7118, 7119, 7127, 7142, 7161, 7180, 7203, 7206, 7208, 7209, 7216, 7224, 7228, 7248, 7279, 7290, 7302, 7334, 7338, 7345, 7352, 7355, 7357, 7358, 7359, 7365, 7382, 7384, 7431, 7441, 7446, 7455, 7469, 7476, 7480, 7492, 7497, 7505, 7510, 7513, 7519, 7537, 7541, 7557, 7564, 7565, 7570, 7573, 7576, 7578, 7592, 7599, 7601, 7604, 7623, 7638, 7647, 7648, 7656, 7658, 7694, 7709, 7711, 7733, 7760, 7789, 7806, 7824, 7831, 7840, 7842, 7849, 7853, 7875, 7888, 7892, 7913, 7923, 7924, 7933, 7934, 7945, 7956, 8014, 8031, 8039, 8076, 8077, 8096, 8098, 8106, 8111, 8140, 8141, 8147, 8148, 8164, 8186, 8205, 8219, 8221, 8238, 8250, 8251, 8259, 8263, 8264, 8265, 8268, 8269, 8275, 8305, 8318, 8320, 8323, 8325, 8334, 8336, 8342, 8343, 8355, 8362, 8373, 8376, 8387, 8401, 8418, 8429, 8441, 8454, 8460, 8473, 8482, 8489, 8491, 8506, 8530, 8556, 8563, 8564, 8567, 8570, 8581, 8590, 8595, 8600, 8601, 8631, 8632, 8663, 8669, 8674, 8676, 8692, 8702, 8711, 8729, 8733, 8771, 8781, 8784, 8787, 8818, 8819, 8838, 8839, 8840, 8845, 8846, 8847, 8848, 8855, 8856, 8858, 8902, 8905, 8906, 8914, 8931, 8935, 8936, 8939, 8965, 8976, 8994, 9000, 9017, 9025, 9029, 9033, 9042, 9060, 9075, 9092, 9114, 9115, 9118, 9122, 9135, 9138, 9139, 9141, 9148, 9151, 9164, 9165, 9167, 9206, 9213, 9249, 9250, 9257, 9263, 9265, 9286, 9291, 9307, 9320, 9328, 9338, 9407, 9415, 9420, 9430, 9443, 9456, 9467, 9471, 9479, 9482, 9488, 9489, 9494, 9496, 9499, 9514, 9517, 9518, 9521, 9527, 9532, 9537, 9548, 9566, 9571, 9572, 9573, 9581, 9597, 9599, 9601, 9611, 9619, 9621, 9635, 9641, 9658, 9713, 9718, 9733, 9739, 9745, 9746, 9747, 9763, 9771, 9787, 9794, 9798, 9806, 9819, 9820, 9835, 9839, 9840, 9860, 9876, 9889, 9914, 9915, 9916, 9922, 9925, 9950, 9956, 9978, 9987, 9991, 9995, 10001, 10002, 10023, 10051, 10052, 10066, 10073, 10103, 10107, 10144, 10146, 10157, 10160, 10161, 10175, 10192, 10199, 10200, 10201, 10202, 10203, 10210, 10233, 10234, 10257, 10268, 10270, 10275, 10289, 10294, 10317, 10319, 10320, 10323, 10334, 10352, 10356, 10357, 10364, 10384, 10445, 10449, 10450, 10473, 10488, 10501, 10508, 10509, 10519, 10533, 10534, 10536, 10537, 10538, 10540, 10541, 10543, 10551, 10562, 10569, 10577, 10580, 10596, 10598, 10623, 10624, 10626, 10640, 10649, 10670, 10674, 10678, 10684, 10690, 10691, 10710, 10715, 10753, 10756, 10757, 10758, 10761, 10775, 10776, 10783, 10788, 10796, 10804, 10806, 10816, 10817, 10832, 10855, 10864, 10865, 10866, 10867, 10874, 10900, 10914, 10919, 10921, 10929, 10930, 10949, 10951, 10974, 10979, 10980, 10986, 10987, 10994, 11014, 11015, 11031, 11038, 11045, 11063, 11082, 11084, 11097, 11132, 11138, 11139, 11157, 11166, 11235, 11258, 11275, 11277, 11293, 11298, 11302, 11308, 11309, 11322, 11327, 11345, 11355, 11358, 11364, 11368, 11378, 11379, 11386, 11388, 11390, 11402, 11404, 11406, 11413, 11417, 11431, 11440, 11441, 11444, 11445, 11446, 11448, 11463, 11486, 11533, 11562, 11568, 11576, 11584, 11588, 11605, 11626, 11627, 11637, 11641, 11652, 11653, 11660, 11672, 11674, 11675, 11686, 11690, 11698, 11710, 11754, 11755, 11772, 11784, 11787, 11803, 11813, 11814, 11855, 11864, 11874, 11883, 11924, 11948, 11960, 11961, 11974, 11979, 11983, 12000, 12015, 12030, 12043, 12046, 12056, 12058, 12060, 12067, 12090, 12094, 12100, 12107, 12110, 12114, 12121, 12122, 12126, 12130, 12136, 12138, 12141, 12142, 12146, 12156, 12157, 12164, 12166, 12167, 12168, 12177, 12182, 12191, 12221, 12232, 12248, 12252, 12259, 12275, 12277, 12285, 12291, 12301, 12317, 12331, 12344, 12375, 12376, 12377, 12383, 12392, 12406, 12423, 12426, 12437, 12452, 12489, 12503, 12506, 12514, 12515, 12522, 12529, 12531, 12557, 12564, 12565, 12567, 12610, 12618, 12626, 12637, 12650, 12655, 12661, 12664, 12665, 12672, 12674, 12679, 12704, 12705, 12707, 12722, 12729, 12733, 12740, 12755, 12756, 12765, 12791, 12811, 12821, 12835, 12845, 12851, 12852, 12853, 12854, 12855, 12872, 12878, 12882, 12893, 12895, 12899, 12903, 12913, 12926, 12941, 12957, 12958, 12961, 12966, 12979, 12997, 12998, 13008, 13032, 13052, 13064, 13065, 13066, 13076, 13092, 13094, 13122, 13140, 13154, 13164, 13166, 13173, 13175, 13185, 13186, 13199, 13200, 13204, 13205, 13206, 13209, 13228, 13229, 13235, 13239, 13252, 13297, 13309, 13312, 13330, 13342, 13350, 13359, 13365, 13387, 13404, 13411, 13412, 13414, 13427, 13428, 13435, 13443, 13460, 13466, 13470, 13472, 13474, 13477, 13484, 13489, 13517, 13526, 13540, 13551, 13555, 13557, 13558, 13559, 13563, 13565, 13567, 13623, 13653, 13654, 13668, 13671, 13678, 13679, 13707, 13724, 13730, 13740, 13743, 13770, 13777, 13780, 13791, 13794, 13800, 13828, 13831, 13837, 13846, 13862, 13870, 13875, 13899, 13901, 13905, 13938, 13952, 13982, 13986, 13992, 13999, 14003, 14004, 14005, 14009, 14016, 14029, 14031, 14049, 14065, 14070, 14104, 14117, 14141, 14142, 14155, 14156, 14166, 14175, 14212, 14213, 14218, 14220, 14222, 14226, 14263, 14270, 14276, 14280, 14281, 14298, 14316, 14322, 14327, 14329, 14339, 14349, 14353, 14384, 14405, 14407, 14414, 14435, 14442, 14447, 14457, 14468, 14506, 14516, 14523, 14528, 14529, 14536, 14544, 14552, 14571, 14575, 14579, 14597, 14603, 14622, 14630, 14648, 14652, 14674, 14700, 14701, 14704, 14706, 14724, 14746, 14756, 14759, 14771, 14773, 14775, 14777, 14790, 14814, 14832, 14833, 14863, 14891, 14897, 14914, 14917, 14921, 14922, 14925, 14927, 14958, 14972, 15003, 15012, 15013, 15032, 15035, 15039, 15048, 15049, 15055, 15061, 15064, 15065, 15067, 15080, 15085, 15114, 15116, 15123, 15129, 15158, 15175, 15207, 15212, 15218, 15233, 15239, 15245, 15254, 15257, 15270, 15274, 15287, 15327, 15330, 15337, 15342, 15347, 15349, 15353, 15356, 15358, 15360, 15367, 15373, 15377, 15379, 15380, 15381, 15390, 15394, 15426, 15432, 15456, 15458, 15543, 15603, 15633, 15638, 15674, 15675, 15681, 15709, 15733, 15742, 15755, 15756, 15757, 15760, 15787, 15805, 15810, 15812, 15832, 15833, 15834, 15835, 15836, 15852, 15857, 15859, 15863, 15896, 15902, 15926, 15938, 15941, 15947, 15955, 15982, 15991, 15994, 15997, 16000, 16010, 16013, 16035, 16050, 16051, 16052, 16053, 16057, 16064, 16065, 16066, 16067, 16079, 16096, 16101, 16109, 16110, 16135, 16137, 16143, 16144, 16145, 16174, 16175, 16194, 16198, 16199, 16201, 16209, 16210, 16216, 16243, 16249, 16252, 16269, 16273, 16274, 16276, 16277, 16278, 16293, 16294]
    # skip = []

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
    if len(sys.argv) != 3:
        print('Usage: <root with data> <path to commits to remove>')
        exit(1)
    root = Path(sys.argv[1])
    commits_to_remove = {l.split(':')[0] for l in Path(sys.argv[2]).read_text().splitlines(keepends=False)}
    root_concat = root.joinpath('pre_train')
    root_concat.mkdir(exist_ok=True)
    concatenated_dataset = PatchNetDataset(root_concat, None, None)
    for folder in sorted(os.listdir(str(root))):
        if folder.startswith('pre_train_'):
            pre_train_root = root.joinpath(folder)
            pre_train_dataset = PatchNetDataset(pre_train_root, None, None)
            pre_train_dataset.load()
            print(f'Size of {pre_train_root}: {len(pre_train_dataset.data_samples)}')
            concatenated_dataset.add(pre_train_dataset)
    concatenated_dataset.filter_empty()
    concatenated_dataset.remove_commits(commits_to_remove)
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


def remove_static_object_from_dataset():
    if len(sys.argv) != 2:
        print('Usage: <root>')
        exit(1)
    root = Path(sys.argv[1])
    dataset = PatchNetDataset(root, None, None)
    dataset.load()
    for sample in dataset.data_samples:
        del sample.commit.prev_updated_generator
    dataset.write_data()


if __name__ == "__main__":
    # cut_dataset(200, shuffle=False)
    # split_on_train_test_val()
    # partition_data()
    # create_k_folds()
    # create_k_folds_for_patchnet()
    # keep_only_intersection_of_commits()
    # convert_to_patchnet_format_list_of_commits()
    # extract_timestamps()
    # mine_dataset()
    # load_dataset()
    # apply_tokenizer_again()
    calculate_performance_metrics_for_patchnet_model()
    # draw_metrics_plots_patchnet_training()
    # extract_changed_files_num()
    # print_changed_files_information()
    # concat_pre_train_datasets()
    # remove_static_object_from_dataset()
    # unite_two_datasets()

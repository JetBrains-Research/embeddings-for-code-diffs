import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, List, Dict

import yaml


class Defects4JFilter:
    def __init__(self, config_file: Path) -> None:
        super().__init__()
        self.config = yaml.safe_load(config_file.open('r'))

    def filter(self, dataset_root: Path, bugs_descriptions_filepath: Path) -> List[Dict[str, Any]]:
        """
        :param bugs_descriptions_filepath: json file which contains description of bugs (for example patterns in each bug)
        :param dataset_root: root where dataset is stored
        :return: list of bugs descriptions which satisfy filtering conditions
        """
        with bugs_descriptions_filepath.open('r') as bugs_descriptions_file:
            bugs_descriptions = json.load(bugs_descriptions_file)
        filtered_bugs_descriptions = \
            list(filter(lambda bug_description: self.should_be_taken(bug_description, dataset_root), bugs_descriptions))
        Defects4JFilter.add_method_pair_roots(filtered_bugs_descriptions, dataset_root)
        return filtered_bugs_descriptions

    @staticmethod
    def add_method_pair_roots(bugs_descriptions: Iterator[Dict[str, Any]], dataset_root: Path) -> None:
        for bug_description in bugs_descriptions:
            project_name = bug_description['project']
            bug_id = bug_description['bugId']
            bug_data_path = dataset_root.joinpath(project_name, str(bug_id))
            abstracted_paths = list(bug_data_path.rglob('abstracted'))
            if len(abstracted_paths) == 0:
                print(f'No abstracted data found for {str(bug_data_path.absolute())}')
            bug_description['method_pair_roots'] = abstracted_paths

    def should_be_taken(self, bug_description: Dict[str, Any], dataset_root: Path) -> bool:
        should_be_taken_bug_description = self.should_be_taken_based_on_bug_description(bug_description)
        should_be_taken_tokens_num = self.should_be_taken_based_on_tokens_num(bug_description, dataset_root)
        return should_be_taken_bug_description and should_be_taken_tokens_num

    def should_be_taken_based_on_tokens_num(self, bug_description: Dict[str, Any], dataset_root: Path) -> bool:
        project_name = bug_description['project']
        bug_id = bug_description['bugId']
        bug_data_path = dataset_root.joinpath(project_name, str(bug_id))
        tokenized_method_paths = list(bug_data_path.rglob('prev.txt')) + list(bug_data_path.rglob('updated.txt'))
        for tokenized_method_path in tokenized_method_paths:
            tokens = tokenized_method_path.read_text().split()
            if len(tokens) > self.config['max_tokens_num']:
                return False
        return True

    def should_be_taken_based_on_bug_description(self, bug_description: Dict[str, Any]) -> bool:
        metrics = bug_description['metrics']
        if metrics['files'] > self.config['max_number_of_changed_files']\
                or metrics['files'] < self.config['min_number_of_changed_files']:
            return False
        if metrics['methods'] > self.config['max_number_of_changed_methods']\
                or metrics['methods'] < self.config['min_number_of_changed_methods']:
            return False
        if metrics['classes'] > self.config['max_number_of_changed_classes']\
                or metrics['classes'] < self.config['min_number_of_changed_classes']:
            return False
        if len(bug_description['repairPatterns']) > self.config['max_number_of_patterns']\
                or len(bug_description['repairPatterns']) < self.config['min_number_of_patterns']:
            return False
        if len(bug_description['repairActions']) > self.config['max_number_of_actions']\
                or len(bug_description['repairActions']) < self.config['min_number_of_actions']:
            return False
        return True


def save_filtered_dataset(bugs_descriptions: List[Dict[str, Any]], output_root: Path) -> None:
    paths_file = output_root.joinpath('paths.txt')
    prev_file = output_root.joinpath('prev.txt')
    updated_file = output_root.joinpath('updated.txt')
    classes_file = output_root.joinpath('classes.txt')

    paths_file_lines = []
    prev_file_lines = []
    updated_file_lines = []
    classes_file_lines = []
    for bug_description in bugs_descriptions:
        patterns = bug_description['repairPatterns']
        for method_root in bug_description['method_pair_roots']:
            paths_file_lines += [str(method_root.absolute())]
            prev_file_lines += [method_root.joinpath('prev.txt').read_text()]
            updated_file_lines += [method_root.joinpath('updated.txt').read_text()]
            classes_file_lines += [str(patterns)]

    paths_file.write_text('\n'.join(paths_file_lines))
    prev_file.write_text('\n'.join(prev_file_lines))
    updated_file.write_text('\n'.join(updated_file_lines))
    classes_file.write_text('\n'.join(classes_file_lines))


def print_statistics(filtered_bugs_descriptions: List[Dict[str, Any]]) -> None:
    print(f'Found {len(filtered_bugs_descriptions)} bugs that satisfy filtering conditions')
    number_of_total_method_pairs = \
        sum(map(lambda bug_desc: len(bug_desc["method_pair_roots"]), filtered_bugs_descriptions))
    print(f'Total number of method pairs: {number_of_total_method_pairs}')
    datapoints_in_each_class = defaultdict(lambda: 0)
    for bug_description in filtered_bugs_descriptions:
        datapoints_in_each_class[', '.join(bug_description["repairPatterns"])] += len(bug_description["method_pair_roots"])
    for pattern in datapoints_in_each_class:
        print(f'{pattern}: {datapoints_in_each_class[pattern]}')


def main() -> None:
    if len(sys.argv) != 5:
        print('Usage: <path to dataset root> <path to json file which describes bugs> '
              '<path to output folder> <path to yml file that describes filtering rules>')
        return
    dataset_root = Path(sys.argv[1])
    bugs_description_filepath = Path(sys.argv[2])
    output_root = Path(sys.argv[3])
    filter_config_file = Path(sys.argv[4])
    defects4j_filter = Defects4JFilter(filter_config_file)
    filtered_bugs_descriptions = defects4j_filter.filter(dataset_root, bugs_description_filepath)
    save_filtered_dataset(filtered_bugs_descriptions, output_root)
    print_statistics(filtered_bugs_descriptions)


if __name__ == "__main__":
    main()

import json
import sys
from pathlib import Path
from typing import Any, Iterator, Tuple, List, Dict

import yaml


class Defects4JFilter:
    def __init__(self, config_file: Path) -> None:
        super().__init__()
        self.config = yaml.safe_load(config_file.open('r'))

    def filter(self, dataset_root: Path, bugs_descriptions_filepath: Path) -> Tuple[List[Path], List[Dict[str, Any]]]:
        """
        :param bugs_descriptions_filepath: json file which contains description of bugs (for example patterns in each bug)
        :param dataset_root: root where dataset is stored
        :return: [
            list of paths where each path represents a directory which contains method pair that satisfies filtering conditions,
            list of bugs descriptions which satisfy filtering conditions
        ]
        """
        with bugs_descriptions_filepath.open('r') as bugs_descriptions_file:
            bugs_descriptions = json.load(bugs_descriptions_file)
        filtered_bugs_descriptions = \
            list(filter(lambda bug_description: self.should_be_taken(bug_description, dataset_root), bugs_descriptions))
        return Defects4JFilter.get_method_pair_roots(filtered_bugs_descriptions, dataset_root), filtered_bugs_descriptions

    @staticmethod
    def get_method_pair_roots(bugs_descriptions: Iterator[Dict[str, Any]], dataset_root: Path) -> List[Path]:
        result = []
        for bug_description in bugs_descriptions:
            project_name = bug_description['project']
            bug_id = bug_description['bugId']
            bug_data_path = dataset_root.joinpath(project_name, str(bug_id))
            abstracted_paths = list(bug_data_path.rglob('abstracted'))
            result += abstracted_paths
        return result

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


def save_filtered_dataset(method_roots: List[Path], output_root: Path) -> None:
    paths_file = output_root.joinpath('paths.txt')
    prev_file = output_root.joinpath('prev.txt')
    updated_file = output_root.joinpath('updated.txt')

    paths_file_lines = []
    prev_file_lines = []
    updated_file_lines = []
    for method_root in method_roots:
        paths_file_lines += [str(method_root.absolute())]
        prev_file_lines += [method_root.joinpath('prev.txt').read_text()]
        updated_file_lines += [method_root.joinpath('updated.txt').read_text()]

    paths_file.write_text('\n'.join(paths_file_lines))
    prev_file.write_text('\n'.join(prev_file_lines))
    updated_file.write_text('\n'.join(updated_file_lines))


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
    filtered_method_roots, filtered_bugs_descriptions = defects4j_filter.filter(dataset_root, bugs_description_filepath)
    save_filtered_dataset(filtered_method_roots, output_root)
    print(f'Found {len(filtered_bugs_descriptions)} bugs that satisfy filtering conditions')
    print(f'Total number of method pairs: {len(filtered_method_roots)}')


if __name__ == "__main__":
    main()

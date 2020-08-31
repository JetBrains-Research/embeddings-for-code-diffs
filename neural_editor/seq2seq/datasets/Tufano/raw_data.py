import sys
from pathlib import Path
from typing import Dict, List
import csv

import numpy as np


def read_csv_raw_data(raw_data_filepath: Path) -> Dict[str, List[str]]:
    result = {'prev': [], 'updated': [], 'class': []}
    with raw_data_filepath.open('r') as raw_data_file:
        input_csv = csv.reader(raw_data_file)
        for input_row in list(input_csv)[1:]:
            result['prev'].append(input_row[0])
            result['updated'].append(input_row[1])
            result['class'].append(input_row[2])
    return result


def sort_by_classes(data_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sorted_classes_positions = np.argsort(list(map(lambda datapoint: str(datapoint), data_dict['class'])))
    result = {'prev': [], 'updated': [], 'class': []}
    for i in sorted_classes_positions:
        result['prev'].append(data_dict['prev'][i])
        result['updated'].append(data_dict['updated'][i])
        result['class'].append(data_dict['class'][i])
    return result


def process_raw_data(raw_data_filepath: Path, output_dir: Path) -> None:
    """
    Writes columns data to separate files in directory specified by output_dir.
    :param raw_data_filepath: csv file with raw data
    :param output_dir: where to save separate files
    :return: nothing
    """
    data_dict = read_csv_raw_data(raw_data_filepath)
    data_dict = sort_by_classes(data_dict)
    prev_filepath = output_dir.joinpath('prev.txt')
    updated_filepath = output_dir.joinpath('updated.txt')
    classes_filepath = output_dir.joinpath('classes.txt')
    prev_filepath.write_text('\n'.join(data_dict['prev']))
    updated_filepath.write_text('\n'.join(data_dict['updated']))
    classes_filepath.write_text('\n'.join(data_dict['class']))


def main():
    if len(sys.argv) != 3:
        print('Usage: <path to raw csv data file with prev, updated, class columns> <output path>')
        return
    raw_data_filepath = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    process_raw_data(raw_data_filepath, output_dir)


if __name__ == "__main__":
    main()

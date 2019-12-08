import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import requests
from tqdm.auto import tqdm


class MdFileParser:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parse(md_path: Path) -> Dict[str, List[Dict[str, str]]]:
        result = defaultdict(list)
        with md_path.open('r') as md_file:
            project_name = None
            for line in md_file:
                if line.startswith('## '):
                    project_name = line[3:-2]
                elif '[Buggy]' in line and '[Fixed]' in line and '[Diff]' in line:
                    i = 0
                    urls = []
                    while i < len(line):
                        if line[i] == '(':
                            url = ''
                            i += 1
                            while line[i] != ')':
                                url += line[i]
                                i += 1
                            urls.append(url)
                        i += 1
                    result[project_name].append({'prev_url': urls[0], 'updated_url': urls[1], 'commit_url': urls[2]})
        return result


class FilesStorage:
    def __init__(self, md_path: Path) -> None:
        super().__init__()
        parsed_dict = MdFileParser.parse(md_path)
        self.files_dict = FilesStorage.download_files(parsed_dict)

    @staticmethod
    def download_files(parsed_dict: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
        result = defaultdict(list)
        for project_name in parsed_dict:
            print()
            print(f'Downloading files from {project_name} project')
            for bug_urls in tqdm(parsed_dict[project_name]):
                bug_files = {
                    'prev_file':
                        FilesStorage.download_file(FilesStorage.get_url_for_raw_source(bug_urls['prev_url'])),
                    'updated_file':
                        FilesStorage.download_file(FilesStorage.get_url_for_raw_source(bug_urls['updated_url']))
                }
                result[project_name].append(bug_files)
        return result

    @staticmethod
    def get_url_for_raw_source(github_file_url: str) -> str:
        url_with_removed_blob = github_file_url.replace('blob/', '')
        return url_with_removed_blob[:8] + 'raw.' + url_with_removed_blob[8:]

    @staticmethod
    def download_file(github_raw_file_url: str) -> str:
        return requests.get(github_raw_file_url).text

    def get_project_names(self) -> List[str]:
        return list(self.files_dict.keys())

    def get_bugs(self, project_name: str) -> List[Dict[str, str]]:
        return self.files_dict[project_name]


class Defects4JCrawler:
    def __init__(self, md_path: Path) -> None:
        super().__init__()
        self.files_storage = FilesStorage(md_path)

    def crawl(self, root_path: Path) -> None:
        root_path = root_path.joinpath('raw_java_files')
        root_path.mkdir(exist_ok=True)
        for project_name in self.files_storage.get_project_names():
            project_path = root_path.joinpath(project_name)
            project_path.mkdir(exist_ok=True)
            for bug_id, bug in enumerate(self.files_storage.get_bugs(project_name), 1):
                bug_path = project_path.joinpath(str(bug_id))
                bug_path.mkdir(exist_ok=True)
                prev_file = bug_path.joinpath('prev.java')
                prev_file.write_text(bug['prev_file'])
                updated_file = bug_path.joinpath('updated.java')
                updated_file.write_text(bug['updated_file'])


def main() -> None:
    if len(sys.argv) != 3:
        print('Usage: '
              '1 argument: path to defects4j-patch.md file from '
              'https://github.com/program-repair/defects4j-dissection\n '
              '2 argument: root where to store output')
        return
    md_file_path: Path = Path(sys.argv[1])
    output_root_path: Path = Path(sys.argv[2])
    Defects4JCrawler(md_file_path).crawl(output_root_path)


if __name__ == "__main__":
    main()

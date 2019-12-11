import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import requests
from requests import Session
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
                    result[project_name].append({
                        'prev_url': urls[0], 'prev_hash': MdFileParser.get_commit_hash(urls[0]),
                        'updated_url': urls[1], 'updated_hash': MdFileParser.get_commit_hash(urls[1]),
                        'owner': MdFileParser.get_owner(urls[2]), 'repo': MdFileParser.get_repo(urls[2]),
                        'commit_url': urls[2]
                    })
        return result

    @staticmethod
    def get_owner(url: str) -> str:
        return url.split('/')[3]

    @staticmethod
    def get_repo(url: str) -> str:
        return url.split('/')[4]

    @staticmethod
    def get_commit_hash(url: str) -> str:
        return url.split('blob/')[1].split('/')[0]


class FilesStorage:
    def __init__(self, md_path: Path, gh_session: Session) -> None:
        super().__init__()
        parsed_dict = MdFileParser.parse(md_path)
        self.files_dict = FilesStorage.download_files(parsed_dict, gh_session)

    @staticmethod
    def get_raw_urls(changed_file_json: Dict[str, Any], bug_data: Dict[str, str]) -> Dict[str, str]:
        slash_split = changed_file_json['raw_url'].split('/')
        result = {
            'prev_raw_url': '/'.join(slash_split[:6] + [bug_data['prev_hash']] + slash_split[7:]),
            'updated_raw_url': '/'.join(slash_split[:6] + [bug_data['updated_hash']] + slash_split[7:])
        }
        return result

    @staticmethod
    def get_raw_urls_of_changed_files(bug_data: Dict[str, str], gh_session: Session) -> List[Dict[str, str]]:
        query_url = f'https://api.github.com/repos/{bug_data["owner"]}/{bug_data["repo"]}/commits/{bug_data["updated_hash"]}'
        query_result = gh_session.get(query_url)
        query_result_json = query_result.json()
        if query_result.status_code != 200 or not query_result.ok:
            raise Exception(f'Cannot retrieve commit data for {bug_data["prev_url"]} and {bug_data["updated_url"]}\n'
                            f'Message: {query_result_json["message"]}')
        changed_files = query_result_json['files']
        return [FilesStorage.get_raw_urls(changed_file_json, bug_data) for changed_file_json in changed_files]

    @staticmethod
    def download_files(parsed_dict: Dict[str, List[Dict[str, str]]], gh_session: Session) -> Dict[str, List[List[Dict[str, str]]]]:
        result = defaultdict(list)
        for project_name in parsed_dict:
            print()
            print(f'Downloading files from {project_name} project')
            for bug_data in tqdm(parsed_dict[project_name]):
                changed_files = FilesStorage.get_raw_urls_of_changed_files(bug_data, gh_session)
                bug_files = []
                for changed_file in changed_files:
                    bug_file = {
                        'prev_file':
                            FilesStorage.download_file(changed_file['prev_raw_url']),
                        'updated_file':
                            FilesStorage.download_file(changed_file['updated_raw_url'])
                    }
                    bug_files.append(bug_file)
                result[project_name].append(bug_files)
        return result

    @staticmethod
    def get_url_for_raw_source(github_file_url: str) -> str:
        url_with_removed_blob = github_file_url.replace('blob/', '')
        return url_with_removed_blob[:8] + 'raw.' + url_with_removed_blob[8:]

    @staticmethod
    def download_file(github_raw_file_url: str) -> str:
        response = requests.get(github_raw_file_url)
        if response.status_code != 200 or not response.ok:
            raise Exception(f'Cannot download file {github_raw_file_url}\n Reason: {response.reason}')
        return response.text

    def get_project_names(self) -> List[str]:
        return list(self.files_dict.keys())

    def get_bugs(self, project_name: str) -> List[List[Dict[str, str]]]:
        return self.files_dict[project_name]


class Defects4JCrawler:
    def __init__(self, md_path: Path, gh_session: Session) -> None:
        super().__init__()
        self.files_storage = FilesStorage(md_path, gh_session)

    def crawl(self, root_path: Path) -> None:
        root_path = root_path.joinpath('raw_java_files')
        root_path.mkdir(exist_ok=True)
        for project_name in self.files_storage.get_project_names():
            project_path = root_path.joinpath(project_name)
            project_path.mkdir(exist_ok=True)
            for bug_id, bug in enumerate(self.files_storage.get_bugs(project_name), 1):
                bug_path = project_path.joinpath(str(bug_id))
                bug_path.mkdir(exist_ok=True)
                for file_id, file in enumerate(bug):
                    file_path = bug_path.joinpath(str(file_id))
                    file_path.mkdir(exist_ok=True)
                    prev_file = file_path.joinpath('prev.java')
                    prev_file.write_text(file['prev_file'])
                    updated_file = file_path.joinpath('updated.java')
                    updated_file.write_text(file['updated_file'])


def authorize(token: str) -> Session:
    gh_session = requests.Session()
    gh_session.auth = ('', token)
    login = gh_session.get('https://api.github.com/user')
    login_json = login.json()
    if login.status_code == 200 and login.ok:
        print(f'Logged in successfully as {login_json["login"]}')
    else:
        print(f'Error occurred during log in. Program will continue without authorization.\n'
              f'Reason: {login.reason}\nMessage: {login_json["message"]}\nStatus code: {login.status_code}')
    return gh_session


def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('Usage: '
              '1 argument: path to defects4j-patch.md file from '
              'https://github.com/program-repair/defects4j-dissection\n'
              '2 argument: root where to store output\n'
              '3 argument(optional): token from GitHub to authorize'
              '(if provided 5000 bugs can be crawled otherwise up to 60)')
        return
    gh_session = None
    if len(sys.argv) == 4:
        gh_session = authorize(sys.argv[3])
    md_file_path: Path = Path(sys.argv[1])
    output_root_path: Path = Path(sys.argv[2])
    Defects4JCrawler(md_file_path, gh_session).crawl(output_root_path)


if __name__ == "__main__":
    main()

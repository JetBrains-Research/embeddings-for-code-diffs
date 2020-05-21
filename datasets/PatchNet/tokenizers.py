from typing import List

import pygments
from pygments.lexers.c_cpp import CLexer
from pygments.token import Token


class NaiveCTokenizer(object):
    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, source_code: str) -> List[str]:
        return source_code.split()


class PygmentsCTokenizer(object):
    # TODO: ignore comments?
    BANNED_TOKEN_TYPES = [Token.Text, Token.Comment, Token.Comment.Multiline, Token.Comment.Single]

    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, source_code: str) -> List[str]:
        lexer = pygments.lex(source_code, CLexer())
        tokens = [t for t in lexer]
        filtered_tokens = [token[1].strip().replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()
                           for token in tokens if token[0] not in PygmentsCTokenizer.BANNED_TOKEN_TYPES]
        return filtered_tokens

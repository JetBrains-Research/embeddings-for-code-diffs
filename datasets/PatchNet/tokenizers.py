from collections import Counter
from typing import List, Tuple, Any

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

    def tokenize(self, source_code: str) -> Tuple[List[str], Counter]:
        lexer = pygments.lex(source_code, CLexer())
        tokens = [t for t in lexer]
        filtered_tokens = [token[1].strip().replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()
                           for token in tokens if not self.in_banned_token_types(token[0], token[1])]
        return filtered_tokens, self.count_identifier_names(tokens)

    def count_identifier_names(self, tokens: List[Tuple[str, Any]]) -> Counter:
        return Counter([token[1] for token in tokens if token[0] in Token.Name])

    def in_banned_token_types(self, token_type, token: str) -> bool:
        if token_type in Token.Text:
            return True
        elif token_type in Token.Comment:
            return True
        elif token_type in Token.Literal and token not in ['0', '1']:
            return True
        else:
            return False

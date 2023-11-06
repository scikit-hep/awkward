# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ast
import sys
from typing import Iterator, NamedTuple


class Flake8ASTErrorInfo(NamedTuple):
    line_number: int
    offset: int
    msg: str
    cls: type  # unused


class Visitor(ast.NodeVisitor):
    msg = "AK101 exception must be wrapped in ak._errors.*_error"

    def __init__(self):
        self.errors: list[Flake8ASTErrorInfo] = []

    def visit_Raise(self, node):
        if isinstance(node.exc, ast.Call):
            if isinstance(node.exc.func, ast.Attribute):
                if node.exc.func.attr in {"wrap_error", "index_error"}:
                    return self.generic_visit(node)
            elif isinstance(node.exc.func, ast.Name):
                if node.exc.func.id in {"wrap_error", "index_error", "ImportError"}:
                    return self.generic_visit(node)

        self.errors.append(
            Flake8ASTErrorInfo(node.lineno, node.col_offset, self.msg, type(self))
        )


class AwkwardASTPlugin:
    name = "flake8_awkward"
    version = "0.0.0"

    def __init__(self, tree: ast.AST) -> None:
        self._tree = tree

    def run(self) -> Iterator[Flake8ASTErrorInfo]:
        visitor = Visitor()
        visitor.visit(self._tree)
        yield from visitor.errors


def main(path):
    with open(path) as f:
        code = f.read()

    node = ast.parse(code)
    plugin = AwkwardASTPlugin(node)
    for err in plugin.run():
        print(f"{path}:{err.line_number}:{err.offset} {err.msg}")


if __name__ == "__main__":
    for item in sys.argv[1:]:
        main(item)

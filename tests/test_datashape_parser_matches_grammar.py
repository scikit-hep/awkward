# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest

pytest.importorskip("lark")

ROOT = Path(__file__).parents[1]
GRAMMAR = "studies/type-parser-2/awkward_datashape_grammar.g"
PARSER = "src/awkward/types/_awkward_datashape_parser.py"
COMMAND = (
    "uv run --no-project --with lark python -m lark.tools.standalone "
    f"{GRAMMAR} -o {PARSER}"
)


def test_parser_matches_grammar():
    committed = (ROOT / PARSER).read_text(encoding="utf-8")
    generated = subprocess.run(
        [sys.executable, "-m", "lark.tools.standalone", str(ROOT / GRAMMAR)],
        capture_output=True,
        check=True,
        encoding="utf-8",
    ).stdout
    assert _normalize(committed) == _normalize(generated), (
        f"{PARSER} does not match {GRAMMAR}; regenerate it with\n\n    {COMMAND}"
    )


def _normalize(source):
    """Return `source` with the parse table in `DATA` renumbered canonically.

    The standalone tool numbers the parser states and tokens differently
    on each run, so two outputs of the same grammar differ only there.
    """
    lines = source.splitlines()
    i = lines.index("DATA = (") + 1
    data = ast.literal_eval(lines[i])
    data["parser"]["parser"] = _canonical_table(data["parser"]["parser"])
    return lines[:i], data, lines[i + 1 :]


def _canonical_table(table):
    """Return `table` with tokens by name and states in breadth-first order."""
    tokens = table["tokens"]
    states = table["states"]
    order = {}
    queue = deque(table["start_states"][name] for name in sorted(table["start_states"]))
    seen = set(queue)
    while queue:
        state = queue.popleft()
        order[state] = len(order)
        for token in sorted(states[state], key=tokens.__getitem__):
            action, arg = states[state][token]
            if action == 0 and arg not in seen:  # Shift: arg is a state
                seen.add(arg)
                queue.append(arg)
    return {
        "states": {
            order[state]: {
                tokens[token]: (action, order[arg] if action == 0 else arg)
                for token, (action, arg) in row.items()
            }
            for state, row in states.items()
        },
        "start_states": {k: order[v] for k, v in table["start_states"].items()},
        "end_states": {k: order[v] for k, v in table["end_states"].items()},
    }

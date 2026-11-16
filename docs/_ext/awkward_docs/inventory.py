# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""``awkward.*`` aliases in the intersphinx inventory, ``objects.inv``."""

import zlib

# objects.inv starts with four uncompressed header lines
_HEADER_LINES = 4


def add_awkward_aliases(inventory: bytes) -> bytes:
    """Return the inventory with an ``awkward.*`` copy of each ``ak.*`` entry.

    Downstream projects that annotate ``awkward.Array`` (resolved from
    ``import awkward as ak``) need ``awkward.*`` entries to link to these
    docs through intersphinx; see
    https://github.com/scikit-hep/awkward/issues/3950. An inventory without
    ``ak.*`` entries is returned unchanged.
    """
    *header_lines, compressed = inventory.split(b"\n", _HEADER_LINES)
    header = b"".join(line + b"\n" for line in header_lines)
    body = zlib.decompress(compressed).decode("utf-8")

    aliases = [
        "awkward." + line[3:] for line in body.splitlines() if line.startswith("ak.")
    ]
    if not aliases:
        return inventory
    new_body = body.rstrip("\n") + "\n" + "\n".join(aliases) + "\n"
    return header + zlib.compress(new_body.encode("utf-8"))

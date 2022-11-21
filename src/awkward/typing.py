# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    # from typing_extensions import Self
    Self = typing.Any
else:
    TypeAlias = object
    Self = object

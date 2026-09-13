# ak.prettyprint.Formatter

Defined in [awkward.prettyprint](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/prettyprint.py) on [line 285](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/prettyprint.py#L285).

#### *class* ak.prettyprint.Formatter(formatters: [FormatterOptions](sphinx-llm:34dbd3f48e6145ea9cc78c1fbc638026#ak.prettyprint.FormatterOptions) | [None](https://docs.python.org/3/library/constants.html#None) = None, precision: [int](https://docs.python.org/3/library/functions.html#int) = 3)

#### \_formatters *: [FormatterOptions](sphinx-llm:34dbd3f48e6145ea9cc78c1fbc638026#ak.prettyprint.FormatterOptions)*

#### \_precision *: [int](https://docs.python.org/3/library/functions.html#int)* *= 3*

#### \_cache *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[type](https://docs.python.org/3/library/functions.html#type), FormatterType]*

#### \_\_call_\_(obj: awkward._typing.Any) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_format_complex(data: [complex](https://docs.python.org/3/library/functions.html#complex)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_format_real(data: [float](https://docs.python.org/3/library/functions.html#float)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

#### \_find_formatter_impl(cls: [type](https://docs.python.org/3/library/functions.html#type)) → FormatterType

# How to convert to/from JSON

Any JSON data can be converted to Awkward Arrays and any Awkward Arrays can be converted to JSON. Awkward type information, such as the distinction between fixed-size and variable-length lists, is lost in the transformation to JSON, however.

```ipython3
import awkward as ak
import pathlib
```

## From JSON to Awkward

The function for JSON → Awkward conversion is [`ak.from_json()`](sphinx-llm:8dc434abe56445a0ba5c48a99f18b65e#ak.from_json).

It can be given a JSON string:

```ipython3
ak.from_json("[[1.1, 2.2, 3.3], [], [4.4, 5.5]]")
```

or a file name:

```ipython3
!echo "[[1.1, 2.2, 3.3], [], [4.4, 5.5]]" > /tmp/awkward-example-1.json
```

```ipython3
ak.from_json(pathlib.Path("/tmp/awkward-example-1.json"))
```

If the dataset contains a single JSON object, an [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) is returned, rather than an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array).

```ipython3
ak.from_json('{"x": 1, "y": [1, 2], "z": "hello"}')
```

## From Awkward to JSON

The function for Awkward → JSON conversion is [`ak.to_json()`](sphinx-llm:7af8124fc8d14a3caf95668155d7904d#ak.to_json).

With one argument, it returns a string.

```ipython3
ak.to_json(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
```

But if a `destination` is given, it is taken to be a filename for output.

```ipython3
ak.to_json(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]), "/tmp/awkward-example-2.json")
```

```ipython3
!cat /tmp/awkward-example-2.json
```

```myst-ansi
[[1.1,2.2,3.3],[],[4.4,5.5]]
```

## Conversion of different types

All of the rules that apply for Python objects in [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) and [`ak.to_list()`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) apply to [`ak.from_json()`](sphinx-llm:8dc434abe56445a0ba5c48a99f18b65e#ak.from_json) and [`ak.to_json()`](sphinx-llm:7af8124fc8d14a3caf95668155d7904d#ak.to_json), replacing builtin Python types for JSON types. (One exception: JSON has no equivalent of a Python tuple.)

## Performance

Since Awkward Array internally uses [RapidJSON](https://rapidjson.org/) to simultaneously parse and convert the JSON string, [`ak.from_json()`](sphinx-llm:8dc434abe56445a0ba5c48a99f18b65e#ak.from_json) and [`ak.to_json()`](sphinx-llm:7af8124fc8d14a3caf95668155d7904d#ak.to_json) should always be faster and use less memory than [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) and [`ak.to_list()`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list). Don’t convert JSON strings into or out of Python objects for the sake of converting them as Python objects: use the JSON converters directly.

# ak.from_avro_file

Defined in [awkward.operations.ak_from_avro_file](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_avro_file.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_avro_file.py#L16).

#### ak.from_avro_file(file, limit_entries=None, \*, debug_forth=False, highlevel=True, behavior=None, attrs=None)

Reads an Avro file as an Awkward Array.

Internally this function uses AwkwardForth DSL. The function recursively
parses the Avro schema, generates Awkward form and Forth code for that
specific Avro file and then reads it.

* **Parameters:**
  * **file** (*path-like* *or* *file-like object*) – Avro file to be read as Awkward Array.
  * **limit_entries** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The number of rows of the Avro file to be read into the Awkward Array.
  * **debug_forth** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, prints the generated Forth code for debugging.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) read from the given Avro file.

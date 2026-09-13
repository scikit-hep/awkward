# ak.to_layout

Defined in [awkward.operations.ak_to_layout](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_layout.py) on [line 30](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_layout.py#L30).

#### ak.to_layout(array, \*, allow_record=True, allow_unknown=False, none_policy='error', use_from_iter=True, primitive_policy='promote', string_policy='as-characters', regulararray=True)

Converts data into a low-level layout object.

This function is usually used to sanitize inputs for other functions; it
would rarely be used in a data analysis because [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) and
[`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) are lower-level than [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array).

* **Parameters:**
  * **array** – Array-like data. May be a high level [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array), [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) (if `allow_record`),
    [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder), or low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) (if `allow_record`),
    or a supported backend array (NumPy `ndarray`, CuPy `ndarray`,
    JAX Array), data-less TypeTracer, Arrow object, or an arbitrary Python
    iterable (for [`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) to convert).
  * **allow_record** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, allow [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) as an output;
    otherwise, if the output would be a scalar record, raise an error.
  * **allow_unknown** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, allow non-Awkward outputs; otherwise,
    raise an error.
  * **use_from_iter** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, allow conversion of iterable inputs to
    arrays using [`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter); otherwise, throw an Exception.
  * **none_policy** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – (“error”, “pass-through”, “promote”): If “error”, throw an Exception
    for None inputs; if “pass-through”, return None; otherwise,
    for “promote” return a length-one array containing the None with unknown type.
  * **primitive_policy** ( *"error"* *,*  *"pass-through"* *,*  *"promote"*) – If “error”, throw an Exception
    for scalar inputs; if “pass-through”, return the scalar input; otherwise,
    for “promote” return a length-one array containing the scalar.
  * **string_policy** ( *"error"* *,*  *"pass-through"* *,*  *"as-characters"* *,*  *"promote"*) – If “error”, throw an Exception
    for scalar inputs; if “pass-through”, return the scalar input;
    ir “as-characters”, return an array of characters; otherwise,
    for “promote” return a length-one array containing the string.
  * **regulararray** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Prefer to create [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) nodes for
    regular array objects.
* **Returns:**
  A low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) (or scalar) holding the data of `array`.

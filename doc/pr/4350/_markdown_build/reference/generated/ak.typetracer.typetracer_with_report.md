# ak.typetracer.typetracer_with_report

Defined in [awkward.typetracer](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/typetracer.py) on [line 177](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/typetracer.py#L177).

#### ak.typetracer.typetracer_with_report(form: awkward.forms.Form | [str](https://docs.python.org/3/library/stdtypes.html#str) | [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping), \*, buffer_key: [str](https://docs.python.org/3/library/stdtypes.html#str) | [collections.abc.Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) = '{form_key}', highlevel: [bool](https://docs.python.org/3/library/functions.html#bool) = False, behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None) = None, attrs: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[awkward.contents.Content, awkward._nplikes.typetracer.TypeTracerReport]

* **Parameters:**
  * **form** ([`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form) or str/dict equivalent) – The form of the Awkward
    Array to build a typetracer-backed array from.
  * **buffer_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *callable*) – Python format string containing
    `"{form_key}"` and/or `"{attribute}"` or a function that takes these
    as keyword arguments and returns a string to use as a `form_key`
    for low-level typetracer buffers.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Returns a typetracer array and associated report object built from a form
with labelled form keys.

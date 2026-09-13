# ak.angle

Defined in [awkward.operations.ak_angle](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_angle.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_angle.py#L16).

#### ak.angle(val, deg=False, highlevel=True, behavior=None, attrs=None)

Returns the counterclockwise angle of each complex element in radians or degrees.

* **Parameters:**
  * **val** – array_like
    Input array.
  * **deg** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is False*) – If True, returns angles in degrees,
    otherwise in radians.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  The counterclockwise angle from the positive real axis on the complex
  plane in the range `(-pi, pi]`, with dtype as a float.

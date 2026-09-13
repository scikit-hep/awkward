# ak.mixin_class_method

Defined in [awkward.behaviors.mixins](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/behaviors/mixins.py) on [line 73](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/behaviors/mixins.py#L73).

#### ak.mixin_class_method(ufunc, rhs=None, \*, transpose=True)

* **Parameters:**
  * **ufunc** ([*numpy.ufunc*](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html#numpy.ufunc)) – A universal function (or NEP18 callable) that is
    hooked in Awkward Array, i.e. it can be the first argument of a behavior.
  * **rhs** (*Set* *[*[*type*](https://docs.python.org/3/library/functions.html#type) *] or* *None*) – Set of right-hand side argument types, optional
    if wrapping a unary function. The left-hand side is expected to
    always be `self` of the parent class. The current class is implicitly
    included in this set.
  * **transpose** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If true, automatically create a transpose signature
    (only makes sense for binary ufuncs).

This decorator can be used to register a mixin class method.

Using this decorator ensures that derived classes that are declared with the
[`ak.mixin_class`](sphinx-llm:b538e0254b654351b7f44e93f2679d36#ak.mixin_class) decorator will also have the behaviors that this class has.

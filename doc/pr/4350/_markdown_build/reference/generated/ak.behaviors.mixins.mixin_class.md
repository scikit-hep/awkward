# ak.behaviors.mixins.mixin_class

Defined in [awkward.behaviors.mixins](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/behaviors/mixins.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/behaviors/mixins.py#L12).

#### ak.behaviors.mixins.mixin_class(registry, name=None)

* **Parameters:**
  * **registry** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – The destination behavior mapping registry. Typically,
    this would be the global registry [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior), but one may wish
    to register methods in an alternative way.
  * **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The name to assign to the behaviour class.

This decorator can be used to register a behavior mixin class.

Any inherited behaviors will automatically be made available to the decorated
class.

See the “Mixin decorators” section of [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for further details.

# ak.jax.register_behavior_class

Defined in [awkward.jax](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/jax.py) on [line 86](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/jax.py#L86).

#### ak.jax.register_behavior_class(cls: HighLevelType)

* **Parameters:**
  **cls** – behavior class to register with JAX

Register the behavior class with JAX, if JAX integration is enabled. Otherwise,
queue the type for subsequent registration when/if JAX is registered.

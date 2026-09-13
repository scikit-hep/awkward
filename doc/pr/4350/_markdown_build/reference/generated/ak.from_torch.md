# ak.from_torch

Defined in [awkward.operations.ak_from_torch](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_torch.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_torch.py#L11).

#### ak.from_torch(array)

Converts a PyTorch Tensor into an Awkward Array.

The data is not copied: a CPU tensor shares its buffer with the Awkward
Array, and a CUDA tensor’s buffer is shared through DLPack.

If `array` contains any other data types the function raises an error.

* **Parameters:**
  **array** – (PyTorch Tensor):
  Tensor to convert into an Awkward Array.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given PyTorch tensor.

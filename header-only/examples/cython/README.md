# Cython header-only example

This example demonstrates the use of Cython to build Awkward Arrays using `LayoutBuilder`
and return them to a Python caller. The important files are:
1. `demo_impl.cpp` — an example C++ source file that builds an Awkward Array using `LayoutBuilder`.
2. `include/demo_impl.h` — the corresponding header file associated with `demo_impl.cpp`.
3. `_demo.pyx` — a Cython module that interfaces `demo_impl.cpp` with Python.
4. `_demo_impl.pxd` — a Cython declaration file that declares the C++ types of `demo_impl.cpp` to Cython.

The remaining files are associated with Python/CMake configuration, and are not an important part of this example.

## Usage

1. Install the library
    ```bash
    pip install .
    ```
2. Run the demo function and print the returned `ak.Array`
    ```python
    from demo import create_demo_array

    print(
        create_demo_array()
    )
    ```

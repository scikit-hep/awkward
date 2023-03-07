# Pybind11 header-only example
This example demonstrates the use of pybind11 to build Awkward Arrays using `LayoutBuilder`
and return them to a Python caller. The important files are:
1. `demo.cpp` — an example C++ source file that builds an Awkward Array using `LayoutBuilder`,
   and declares a Python extension module.

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

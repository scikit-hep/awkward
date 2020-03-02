# How to build a C++ project that depends on Awkward

Python code that depends on Awkward requires no configuration; just `pip install awkward1` and import it like any other Python library.

C++ code, however, must have access to Awkward's `include` directory and its static or dynamically linked libraries. When you `pip install awkward1` with superuser privileges (i.e. not `--user`), these files are deployed to your system's standard `include` and `lib` directories (on MacOS and Linux, but not Windows).

The provided libraries are:

   * libawkward.so (or .dylib)
   * libawkward-cpu-kernels.so (or .dylib)
   * libawkward.a
   * libawkward-cpu-kernels.a

You need to link against both libawkward and libawkward-cpu-kernels, but either the shared libraries (.so or .dylib) or the static libraries—not both.

Assuming that `$PREFIX` has your system's standard `$PREFIX/include` and `$PREFIX/lib`, the [minimal.cpp](./minimal.cpp) project can be compiled like this:

```bash
g++ --std=c++11 -I$PREFIX/include -L$PREFIX/lib -lawkward -lawkward-cpu-kernels minimal.cpp -o minimal
```

and used like this (it picks one item from an Awkward Array):

```bash
./minimal "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]]" -2
# [4.4,5.5]
```

If you want your C++ code to have a Python interface, compile it with the same version of pybind11 as Awkward (see release notes) as shown in [CMakeLists.txt](CMakeLists.txt) and [through-python.cpp](through-python.cpp), which can be compiled like this:

```bash
cmake -S . -B build
cmake --build build
```

Its Python test, [test-python.py](test-python.py),

```bash
pytest -vv test-python.py
```

demonstrates that you can pass Awkward Arrays between C++ codebases through Python as an intermediary. The data are not copied or serialized in this transfer; this is a suitable way to move large, complex datasets.

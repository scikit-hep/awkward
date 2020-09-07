# How to build a C++ project that depends on Awkward

Python code that depends on Awkward requires no configuration; just `pip install awkward1` and import it like any other Python library.

C++ compilers, on the other hand, don't know where to find Awkward Array's header files and shared libraries without help. These files are distributed along with the Python library, and it can be used as a "pkg-config" substitute to locate these libraries:

```bash
$ python -m awkward1.config --cflags --libs
-std=c++11 -I/path/to/awkward1/include -L/path/to/awkward1 -lawkward -lawkward-cpu-kernels
```

The Python package includes both static (.a) and shared (.so or .dylib) libraries.

A C++ program that depends on Awkward Array, like [minimal.cpp](./minimal.cpp), can be compiled like this:

```bash
$ g++ `python -m awkward1.config --cflags --libs` minimal.cpp -o minimal
```

The new executable, `./minimal`, depends on Awkward Array's shared libraries, we they need to be on your system's library search path. The way to do that is system-dependent, but a quick and dirty way to do it on Linux is

```bash
export LD_LIBRARY_PATH=`python -m awkward1.config --libdir`:$LD_LIBRARY_PATH
```

Then you can run this executable. (It takes two arguments, an array as JSON and an item to select and print as JSON.)

```bash
$ ./minimal "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]]" -2
[4.4,5.5]
```

If you want your C++ code to have a Python interface, compile it with the same version of pybind11 as Awkward (see release notes) as shown in [CMakeLists.txt](CMakeLists.txt) and [dependent.cpp](dependent.cpp), which can be compiled like this:

```bash
$ cmake -S . -B build
$ cmake --build build
```

Its Python test, [test-python.py](test-python.py),

```bash
$ pytest -vv test-python.py
```

demonstrates that you can pass Awkward Arrays between C++ codebases through Python as an intermediary. The data are not copied or serialized in this transfer; this is a suitable way to move large, complex datasets.

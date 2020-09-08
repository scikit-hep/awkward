# How to build a C++ project that depends on Awkward

Python code that depends on Awkward requires no configuration; just `pip install awkward1` and import it like any other Python library.

C++ compilers, on the other hand, don't know where to find Awkward Array's header files and libraries without help. These files are distributed inside of the Python library. To find the right search paths, `python -m awkward1.config` can be used like "pkg-config":

```bash
$ python -m awkward1.config --cflags --libs
-std=c++11 -I/path/to/awkward1/include -L/path/to/awkward1 -lawkward -lawkward-cpu-kernels
```

The Python package includes both shared (.so or .dylib) and static (.a) libraries (see `python -m awkward1.config --help`).

A C++ program that depends on Awkward Array, such as [minimal.cpp](./minimal.cpp), can be compiled with the following (be sure to configure Awkward libraries *after* your own project's files! argument order matters!):

```bash
$ g++ minimal.cpp `python -m awkward1.config --cflags --libs` -o minimal
```

If you want Awkward Array to be statically linked into the executable, use `--static-libs`. Otherwise, you will either need to copy the shared libraries into a directory on your system's library search path or point your system's library search path to the Awkward package. A quick and dirty way to do the latter on Linux is

```bash
$ export LD_LIBRARY_PATH=`python -m awkward1.config --libdir`:$LD_LIBRARY_PATH
```

Now you can run this executable. (It takes two arguments, an array as JSON and an item to select and print as JSON.)

```bash
$ ./minimal "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]]" -2
[4.4,5.5]
```

If you want your C++ code to have a Python interface, compile it with the same version of pybind11 as Awkward (see release notes) as shown in [CMakeLists.txt](CMakeLists.txt) and [dependent.cpp](dependent.cpp), which can be compiled like the following (you may need to [tell pybind11 which Python to use](https://pybind11.readthedocs.io/en/stable/faq.html#cmake-doesn-t-detect-the-right-python-version)):

```bash
$ cmake -S . -B build
$ cmake --build build
```

Its Python test, [test-python.py](test-python.py),

```bash
$ pytest -vv test-python.py
```

demonstrates that you can pass Awkward Arrays between C++ codebases through Python as an intermediary. The data are not copied or serialized in this transfer; this is a suitable way to move large, complex datasets.

# How to build a C++ project that depends on Awkward

Python code that depends on Awkward requires no configuration; just `pip install awkward1` and use it as any other Python library.

C++ code, however, must have access to Awkward's `include` directory and its static or dynamically linked libraries. When you `pip install awkward1` with superuser privileges (i.e. not `--user`), these files are deployed to your system's standard `include` and `lib` directories (on MacOS and Linux, but not Windows).

The small project in this directory is an example of how to use Awkward as a C++ dependency. The CMake file assumes that 

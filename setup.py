# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import glob
import os
import platform
import subprocess
import sys
import distutils.util
import multiprocessing
import shutil

import setuptools
import setuptools.command.build_ext
import setuptools.command.install

from setuptools import setup, Extension

install_requires = open("requirements.txt").read().strip().split()

extras = {"test": open("requirements-test.txt").read().strip().split(),
          "dev":  ['numba>=0.50.0;python_version>="3.6"',
                   'pandas>=0.24.0;python_version>="3.6"',
                   'numexpr;python_version>="3.6"',
                   'autograd;python_version>="3.6"',
                   'pyarrow>=1.0;python_version>="3.6" and sys_platform != "win32"']}
extras["all"] = sum(extras.values(), [])

tests_require = extras["test"]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(x.name for x in self.extensions))

        for x in self.extensions:
            self.build_extension(x)

    def build_extension(self, ext):

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_INSTALL_PREFIX={0}".format(extdir),
                      "-DPYTHON_EXECUTABLE={0}".format(sys.executable),
                      "-DPYBUILD=ON",
                      "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9",
                      "-DPYBUILD=ON",
                      "-DBUILD_TESTING=OFF"]
        try:
           compiler_path = self.compiler.compiler_cxx[0]
           cmake_args.append("-DCMAKE_CXX_COMPILER={0}".format(compiler_path))
        except AttributeError:
            print("Not able to access compiler path (on Windows), using CMake default")

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{0}={1}".format(cfg.upper(), extdir),
                           "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        if platform.system() == "Windows":
            build_args += ["/m"]
        else:
            build_args += ["-j", str(multiprocessing.cpu_count())]

        if not os.path.exists(self.build_temp):
             os.makedirs(self.build_temp)
        build_dir = self.build_temp

        # for scikit-build:
        # build_dir = os.path.join(DIR, "_pybuild")

        subprocess.check_call(["cmake", "-S", ext.sourcedir, "-B", build_dir] + cmake_args)
        subprocess.check_call(["cmake", "--build", build_dir] + build_args)
        subprocess.check_call(["cmake", "--build", build_dir, "--target", "install"])

if platform.system() == "Windows":
    # Libraries are not installed system-wide on Windows, but they do have to be in the awkward1 directory.
    libraries = []

    def tree(x):
        print(x + (" (dir)" if os.path.isdir(x) else ""))
        if os.path.isdir(x):
            for y in os.listdir(x):
                tree(os.path.join(x, y))

    class Install(setuptools.command.install.install):
        def run(self):
            print("--- build directory -------------------------------------------")
            tree("build")

            print("--- specifically, the dlldir ----------------------------------")
            dlldir = os.path.join(os.path.join("build", "temp.%s-%d.%d" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])), "Release", "Release")
            tree(dlldir)

            print("--- copying ---------------------------------------------------")
            for x in os.listdir(dlldir):
                if x.startswith("awkward"):
                    print("copying", os.path.join(dlldir, x), "-->", os.path.join(self.build_lib, "awkward1", x))
                    shutil.copyfile(os.path.join(dlldir, x), os.path.join(self.build_lib, "awkward1", x))
            print("--- deleting --------------------------------------------------")

            outerdir = os.path.join(os.path.join("build", "lib.%s-%d.%d" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])))
            for x in os.listdir(outerdir):
                if x.endswith(".pyd"):
                    print("deleting", os.path.join(outerdir, x))
                    os.remove(os.path.join(outerdir, x))

            print("--- begin normal install --------------------------------------")
            setuptools.command.install.install.run(self)

else:
    # Libraries do not exist yet, so they cannot be determined with a glob pattern.
    libdir = os.path.join(os.path.join("build", "lib.%s-%d.%d" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])), "awkward1")
    static = ".a"
    if platform.system() == "Darwin":
        shared = ".dylib"
    else:
        shared = ".so"
    libraries = [("lib", [os.path.join(libdir, "libawkward-cpu-kernels-static" + static),
                          os.path.join(libdir, "libawkward-cpu-kernels" + shared),
                          os.path.join(libdir, "libawkward-static" + static),
                          os.path.join(libdir, "libawkward" + shared)])]

    Install = setuptools.command.install.install

setup(name = "awkward1",
      packages = setuptools.find_packages(where="src"),
      package_dir = {"": "src"},
      include_package_data = True,
      package_data = {"": ["*.dll"]},
      data_files = libraries + [
          ("include/awkward",              glob.glob("include/awkward/*.h")),
          ("include/awkward/cpu-kernels",  glob.glob("include/awkward/cpu-kernels/*.h")),
          ("include/awkward/python",       glob.glob("include/awkward/python/*.h")),
          ("include/awkward/array",        glob.glob("include/awkward/array/*.h")),
          ("include/awkward/builder",      glob.glob("include/awkward/builder/*.h")),
          ("include/awkward/partition",    glob.glob("include/awkward/partition/*.h")),
          ("include/awkward/io",           glob.glob("include/awkward/io/*.h")),
          ("include/awkward/type",         glob.glob("include/awkward/type/*.h"))],
      version = open("VERSION_INFO").read().strip(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "Manipulate JSON-like data with NumPy-like idioms.",
      long_description = """<a href="https://github.com/scikit-hep/awkward-1.0#readme"><img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/logo/logo-300px.png"></a>

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

<table>
  <tr>
    <td width="66%" valign="top">
      <a href="https://awkward-array.org">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-tutorials.png" width="570">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.org">
        How-to tutorials
        </a>
      </b></p>
    </td>
    <td width="33%" valign="top">
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-sphinx.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python API reference
        </a>
      </b></p>
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/panel-doxygen.png" width="268">
      </a>
      <p align="center"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++ API reference
        </a>
      </b></p>
    </td>
  </tr>
</table>

# Motivating example

Given an array of objects with `x`, `y` fields and variable-length nested lists like

```python
array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": {1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

the following slices out the `y` values, drops the first element from each inner list, and runs NumPy's `np.square` function on everything that is left:

```python
output = np.square(array["y", ..., 1:])
```

The result is

```python
[
    [[], [4], [4, 9]],
    [],
    [[4, 9, 16], [4, 9, 16, 25]]
]
```

The equivalent using only Python is

```python
output = []
for sublist in array:
    tmp1 = []
    for record in sublist:
        tmp2 = []
        for number in record["y"][1:]:
            tmp2.append(np.square(number))
        tmp1.append(tmp2)
    output.append(tmp1)
```

Not only is the expression using Awkward Arrays more concise, using idioms familiar from NumPy, but it's much faster and uses less memory.

For a similar problem 10 million times larger than the one above (on a single-threaded 2.2 GHz processor),

   * the Awkward Array one-liner takes **4.6 seconds** to run and uses **2.1 GB** of memory,
   * the equivalent using Python lists and dicts takes **138 seconds** to run and uses **22 GB** of memory.

Speed and memory factors in the double digits are common because we're replacing Python's dynamically typed, pointer-chasing virtual machine with type-specialized, precompiled routines on contiguous data. (In other words, for the same reasons as NumPy.) Even higher speedups are possible when Awkward Array is paired with [Numba](https://numba.pydata.org/).

Our [presentation at SciPy 2020](https://youtu.be/WlnUF3LRBj4) provides a good introduction, showing how to use these arrays in a real analysis.

# Installation

Awkward Array can be installed [from PyPI](https://pypi.org/project/awkward1/) using pip:

```bash
pip install awkward1
```

Most users will get a precompiled binary (wheel) for your operating system and Python version. If not, the above attempts to compile from source.

   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward-1.0/issues). If you have a general "How do I…?" question, we'll answer it as a new [example in the tutorial](https://awkward-array.org/how-to.html).
   * If you have a problem that's too specific to be new documentation or it isn't exclusively related to Awkward Array, it might be more appropriate to ask on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * The [Gitter Scikit-HEP/community](https://gitter.im/Scikit-HEP/community) is a way to get in touch with all Scikit-HEP developers and users.
""",
      long_description_content_type = "text/markdown",
      url = "https://github.com/scikit-hep/awkward-1.0",
      download_url = "https://github.com/scikit-hep/awkward-1.0/releases",
      license = "BSD 3-clause",
      entry_points = {
        "numba_extensions": ["init = awkward1._connect._numba:register"]
      },
      test_suite = "tests",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
      install_requires = install_requires,
      tests_require = extras["test"],
      extras_require = extras,

      # cmake_args=['-DBUILD_TESTING=OFF'],      # for scikit-build
      ext_modules = [CMakeExtension("awkward")], # NOT scikit-build
      cmdclass = {"build_ext": CMakeBuild,       # NOT scikit-build
                  "install": Install},

      classifiers = [
#         "Development Status :: 1 - Planning",
#         "Development Status :: 2 - Pre-Alpha",
#         "Development Status :: 3 - Alpha",
#         "Development Status :: 4 - Beta",
          "Development Status :: 5 - Production/Stable",
#         "Development Status :: 6 - Mature",
#         "Development Status :: 7 - Inactive",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX :: Linux",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ])

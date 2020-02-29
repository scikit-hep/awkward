# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import platform
import re
import subprocess
import sys
import shutil
import glob

import setuptools
from setuptools.command.build_ext import build_ext, get_config_var, libtype
from setuptools import setup, Extension
from setuptools.extension import Library


cpu_kernel_sources = glob.glob("src/cpu-kernels/*.cpp")
libawkward_sources = glob.glob("src/libawkward/**/*.cpp", recursive=True)
layout_sources = glob.glob("src/python/layout/*.cpp")
include_dirs = ['rapidjson/include', 'pybind11/include', 'include']

ext_modules = [
    Library(
        "awkward1.awkward-cpu-kernels", cpu_kernel_sources, include_dirs=include_dirs, language="c++",
    ),
    Library(
        "awkward1.awkward", libawkward_sources, include_dirs=include_dirs, language="c++",
        extra_link_arks=["-lawkward-cpu-kernels"]
    ),
    Extension(
        "awkward1.layout", layout_sources + ['src/python/layout.cpp'],
        include_dirs=include_dirs, language="c++", extra_link_args=['-lawkward']
    ),
    Extension(
        "awkward1.types", ['src/python/types.cpp'], include_dirs=include_dirs, language="c++", extra_link_args=['-lawkward']
    ),
    Extension(
        "awkward1._io", ['src/python/_io.cpp'], include_dirs=include_dirs, language="c++", extra_link_args=['-lawkward']
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    # flags = ['-std=c++14', '-std=c++11']
    flags = ['-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc", "/bigobj"], "unix": [""]}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


    def get_ext_filename(self, fullname):
        filename = super(build_ext, self).get_ext_filename(fullname)
        if fullname in self.ext_map:
            ext = self.ext_map[fullname]
            if isinstance(ext, Library):
                so_ext = get_config_var('EXT_SUFFIX')
                filename = filename[:-len(so_ext)]
                fn, ext = os.path.splitext(filename)
                return self.shlib_compiler.library_filename(fn, libtype) 
            else:
                return super(BuildExt, self).get_ext_filename(fullname)


setup(name = "awkward1",
      packages = setuptools.find_packages(exclude=["tests"]),
      scripts = [],
      data_files = ([("", glob.glob("rapidjson/include/rapidjson/*.h")),
                     ("", glob.glob("rapidjson/include/rapidjson/*/*.h")),
                     (os.path.join("awkward1", "include", "awkward"),                glob.glob("include/awkward/*.h")),
                     (os.path.join("awkward1", "include", "awkward", "cpu-kernels"), glob.glob("include/awkward/cpu-kernels/*.h")),
                     (os.path.join("awkward1", "include", "awkward", "python"),      glob.glob("include/awkward/python/*.h")),
                     (os.path.join("awkward1", "include", "awkward", "array"),       glob.glob("include/awkward/array/*.h")),
                     (os.path.join("awkward1", "include", "awkward", "fillable"),    glob.glob("include/awkward/fillable/*.h")),
                     (os.path.join("awkward1", "include", "awkward", "io"),          glob.glob("include/awkward/io/*.h")),
                     (os.path.join("awkward1", "include", "awkward", "type"),        glob.glob("include/awkward/type/*.h"))]),
      version = open("VERSION_INFO").read().strip(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "Development of awkward 1.0, to replace scikit-hep/awkward-array in 2020.",
      long_description = "",
      long_description_content_type = "text/markdown",
      url = "https://github.com/jpivarski/awkward1",
      download_url = "https://github.com/jpivarski/awkward1/releases",
      license = "BSD 3-clause",
      entry_points = {
        "numba_extensions": ["init = awkward1._numba:register"]
      },
      ext_modules = ext_modules,
      cmdclass = {"build_ext": BuildExt},
      test_suite = "tests",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
      install_requires = open("requirements.txt").read().strip().split(),
      tests_require = open("requirements-test.txt").read().strip().split(),
      zip_safe = False,
      classifiers = [
#         "Development Status :: 1 - Planning",
#         "Development Status :: 2 - Pre-Alpha",
#         "Development Status :: 3 - Alpha",
          "Development Status :: 4 - Beta",
#         "Development Status :: 5 - Production/Stable",
#         "Development Status :: 6 - Mature",
#         "Development Status :: 7 - Inactive",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
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
          ],
      )

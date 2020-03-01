# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import glob

# Future change if skbuild supported:
# from skbuild import setup
import setuptools
# else:
import setuptools.command.build_ext as build_ext
from setuptools import setup
import glob
import subprocess
import platform
import sys

class CMakeExtension(setuptools.Extension):
    def __init__(self, name, sourcedir=""):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(setuptools.command.build_ext.build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(x.name for x in self.extensions))

        for x in self.extensions:
            self.build_extension(x)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        print("--- HENRYIII ---")
        print(extdir)
        print(dir(self))
        print(dir(ext))


        cmake_args = ["-DCMAKE_INSTALL_PREFIX={0}".format(extdir),
                      "-DPYTHON_EXECUTABLE=" + sys.executable,
                      "-DPYBUILD=ON",
                      "-DBUILD_TESTING=OFF",
                      "-GNinja"
                      ]

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

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", "install"], cwd=self.build_temp)




install_requires = open("requirements.txt").read().strip().split()

extras = {
    "test": open("requirements-test.txt").read().strip().split(),
    "docs": open("requirements-docs.txt").read().strip().split(),
    "dev":  open("requirements-dev.txt").read().strip().split()}
extras["all"] = sum(extras.values(), [])

tests_require = extras["test"]

setup(name = "awkward1",
      packages = setuptools.find_packages(where="src"),
      package_dir = {"": "src"},
      data_files = ([("include/awkward",             glob.glob("include/awkward/*.h")),
                     ("include/awkward/cpu-kernels", glob.glob("include/awkward/cpu-kernels/*.h")),
                     ("include/awkward/python",      glob.glob("include/awkward/python/*.h")),
                     ("include/awkward/array",       glob.glob("include/awkward/array/*.h")),
                     ("include/awkward/builder",     glob.glob("include/awkward/builder/*.h")),
                     ("include/awkward/io",          glob.glob("include/awkward/io/*.h")),
                     ("include/awkward/type",        glob.glob("include/awkward/type/*.h"))]),
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
      test_suite = "tests",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
      install_requires = install_requires,
      tests_require = extras["test"],
      extras_require = extras,
      # cmake_args=['-DBUILD_TESTING=OFF'],      # SkBuild
      ext_modules = [CMakeExtension("awkward")], # Not SkBuild
      cmdclass = {"build_ext": CMakeBuild},      # Not SkBuild
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

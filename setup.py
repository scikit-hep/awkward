import os
import platform
import re
import subprocess
import sys

import distutils.version
import setuptools
import setuptools.command.build_ext
from setuptools import setup

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

        if platform.system() == "Windows":
            cmake_version = distutils.version.LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for x in self.extensions:
            self.build_extension(x)

    def build_extension(self, ext):
        extdir = os.path.join(os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))), "awkward1")
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{0}={1}".format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]

        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            # build_args += ["--", "-j8"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)
        subprocess.check_call(["ctest", "--output-on-failure"], cwd=self.build_temp)

setup(name = "awkward1",
      packages = setuptools.find_packages(exclude=["tests"]),
      scripts = [],
      version = open("VERSION_INFO").read().strip(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "Development of awkward 1.0, to replace scikit-hep/awkward-array in 2020.",
      long_description = "",
      url = "https://github.com/jpivarski/awkward1",
      download_url = "https://github.com/jpivarski/awkward1/releases",
      license = "BSD 3-clause",
      ext_modules = [CMakeExtension("awkward")],
      cmdclass = {"build_ext": CMakeBuild},
      test_suite = "tests",
      install_requires = open("requirements.txt").read().strip().split(),
      tests_require = open("test-requirements.txt").read().strip().split(),
      zip_safe = False,
      classifiers = [
          "Development Status :: 1 - Planning",
#         "Development Status :: 2 - Pre-Alpha",
#         "Development Status :: 3 - Alpha",
#         "Development Status :: 4 - Beta",
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

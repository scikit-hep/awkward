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


extras = {
    "cuda": ["awkward1-cuda-kernels==" + open("VERSION_INFO").read().strip()],
    "test": open("requirements-test.txt").read().strip().split("\n"),
    "dev":  open("requirements-dev.txt").read().strip().split("\n"),
}
extras["all"] = sum(extras.values(), [])

install_requires = open("requirements.txt").read().strip().split("\n")


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
        build_args = []
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={0}".format(extdir),
            "-DPYTHON_EXECUTABLE={0}".format(sys.executable),
            "-DPYBUILD=ON",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9",
            "-DPYBUILD=ON",
            "-DBUILD_TESTING=OFF",
        ]
        try:
           compiler_path = self.compiler.compiler_cxx[0]
           cmake_args += ["-DCMAKE_CXX_COMPILER={0}".format(compiler_path)]
        except AttributeError:
            print("Not able to access compiler path, using CMake default")

        cfg = "Debug" if self.debug else "Release"
        build_args += ["--config", cfg]
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{0}={1}".format(cfg.upper(), extdir),
                "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
            ]
            cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
            if sys.maxsize > 2**32 and cmake_generator != "NMake Makefiles" and "Win64" not in cmake_generator:
                cmake_args += ["-A", "x64"]
        else:
            build_args += ["-j", str(multiprocessing.cpu_count())]

        if not os.path.exists(self.build_temp):
             os.makedirs(self.build_temp)
        build_dir = self.build_temp

        subprocess.check_call(["cmake", "-S", ext.sourcedir, "-B", build_dir] + cmake_args)
        subprocess.check_call(["cmake", "--build", build_dir] + build_args)
        subprocess.check_call(["cmake", "--build", build_dir, "--config", cfg, "--target", "install"])


def tree(x):
    print(x + (" (dir)" if os.path.isdir(x) else ""))
    if os.path.isdir(x):
        for y in os.listdir(x):
            tree(os.path.join(x, y))


if platform.system() == "Windows":
    class Install(setuptools.command.install.install):
        def run(self):
            outerdir = os.path.join(os.path.join("build", "lib.%s-%d.%d" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])))

            print("--- this directory --------------------------------------------")
            for x in sorted(os.listdir(".")):
                print(x)

            print("--- build directory -------------------------------------------")
            tree("build")

            print("--- copying includes ------------------------------------------")
            shutil.copytree(os.path.join("include"), os.path.join(outerdir, "awkward1", "include"))

            print("--- outerdir after copy ---------------------------------------")
            tree(outerdir)

            print("--- copying libraries -----------------------------------------")
            dlldir = os.path.join(os.path.join("build", "temp.%s-%d.%d" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])), "Release")
            found = False
            for x in os.listdir(dlldir):
                if x.endswith(".lib") or x.endswith(".exp") or x.endswith(".dll"):
                    print("copying", os.path.join(dlldir, x), "-->", os.path.join(self.build_lib, "awkward1", x))
                    shutil.copyfile(os.path.join(dlldir, x), os.path.join(self.build_lib, "awkward1", x))
                    found = True
            if not found:
                dlldir = os.path.join(dlldir, "Release")
                for x in os.listdir(dlldir):
                    if x.endswith(".lib") or x.endswith(".exp") or x.endswith(".dll"):
                        print("copying", os.path.join(dlldir, x), "-->", os.path.join(self.build_lib, "awkward1", x))
                        shutil.copyfile(os.path.join(dlldir, x), os.path.join(self.build_lib, "awkward1", x))
                        found = True

            print("--- deleting libraries ----------------------------------------")
            for x in os.listdir(outerdir):
                if x.endswith(".pyd"):
                    print("deleting", os.path.join(outerdir, x))
                    os.remove(os.path.join(outerdir, x))

            print("--- begin normal install --------------------------------------")
            setuptools.command.install.install.run(self)

else:
    class Install(setuptools.command.install.install):
        def run(self):
            outerdir = os.path.join(os.path.join("build", "lib.%s-%d.%d" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])))

            print("--- this directory --------------------------------------------")
            for x in sorted(os.listdir(".")):
                print(x)

            print("--- build directory -------------------------------------------")
            tree("build")

            print("--- copying includes ------------------------------------------")
            shutil.copytree(os.path.join("include"), os.path.join(outerdir, "awkward1", "include"))

            print("--- outerdir after copy ---------------------------------------")
            tree(outerdir)

            print("--- begin normal install --------------------------------------")
            setuptools.command.install.install.run(self)


setup(name = "awkward1",
      packages = [
          x
          for x in setuptools.find_packages(where="src")
          if x != "awkward1_cuda_kernels"
      ],
      package_dir = {"awkward1": "src/awkward1"},
      version = open("VERSION_INFO").read().strip(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "Manipulate JSON-like data with NumPy-like idioms.",
      long_description = open("README-pypi.md").read(),
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
      ext_modules = [
          CMakeExtension("awkward"),
      ],
      cmdclass = {
          "build_ext": CMakeBuild,
          "install": Install,
      },
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

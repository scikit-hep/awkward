# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import print_function

import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.install
from setuptools import setup, Extension

if sys.version_info < (3,):
    # Note: never import from distutils before setuptools!
    from distutils.util import get_platform
else:
    from sysconfig import get_platform


try:
    import cmake

    CMAKE = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
except ImportError:
    CMAKE = "cmake"

PYTHON = sys.executable


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


with open("VERSION_INFO") as f:
    VERSION_INFO = f.read().strip()


def read_requirements(name):
    with open(name) as f:
        return f.read().strip().split("\n")


extras = {
    "cuda": ["awkward-cuda-kernels=={0}".format(VERSION_INFO)],
    "test": read_requirements("requirements-test.txt"),
    "dev": read_requirements("requirements-dev.txt"),
}
extras["all"] = sum(extras.values(), [])

install_requires = read_requirements("requirements.txt")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        try:
            subprocess.check_output([CMAKE, "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(x.name for x in self.extensions)
            )

        setuptools.command.build_ext.build_ext.build_extensions(self)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]
        cmake_args += [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
            "-DCMAKE_INSTALL_PREFIX={0}".format(extdir),
            "-DPYTHON_EXECUTABLE={0}".format(sys.executable),
            "-DPYBUILD=ON",
            "-DBUILD_TESTING=OFF",
        ]
        build_args = []

        try:
            compiler_path = self.compiler.compiler_cxx[0]
            cmake_args += ["-DCMAKE_CXX_COMPILER={0}".format(compiler_path)]
        except AttributeError:
            print("Not able to access compiler path, using CMake default")

        if self.compiler.compiler_type == "msvc":
            cmake_args.append("-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE")

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j", str(multiprocessing.cpu_count())]

        if platform.system() == "Darwin":
            if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
                cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

            # Cross-compile support for macOS
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args.append(
                    "-DCMAKE_OSX_ARCHITECTURES:STRING={0}".format(";".join(archs))
                )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            [CMAKE, "-S", ext.sourcedir, "-B", self.build_temp] + cmake_args
        )
        subprocess.check_call([CMAKE, "--build", self.build_temp] + build_args)
        subprocess.check_call([CMAKE, "--install", self.build_temp, "--config", cfg])


def tree(x):
    print(("{} (dir)" if os.path.isdir(x) else "{}").format(x))
    if os.path.isdir(x):
        for y in os.listdir(x):
            tree(os.path.join(x, y))


class BuildPy(setuptools.command.build_py.build_py):
    def run(self):
        # generate include/awkward/kernels.h and src/awkward/_kernel_signatures.py
        subprocess.check_call(
            [PYTHON, os.path.join("dev", "generate-kernel-signatures.py")]
        )

        setuptools.command.build_py.build_py.run(self)


class Install(setuptools.command.install.install):
    def run(self):
        outerdir = os.path.join(
            os.path.join(
                "build",
                "lib.{}-{}.{}".format(
                    get_platform(),
                    sys.version_info[0],
                    sys.version_info[1],
                ),
            )
        )

        print("--- this directory --------------------------------------------")
        for x in sorted(os.listdir(".")):
            print(x)

        print("--- build directory -------------------------------------------")
        tree("build")

        print("--- copying includes ------------------------------------------")
        shutil.copytree(
            os.path.join("include"), os.path.join(outerdir, "awkward", "include")
        )

        print("--- outerdir after copy ---------------------------------------")
        tree(outerdir)

        if platform.system() == "Windows":
            print("--- copying libraries -----------------------------------------")
            dlldir = os.path.join(
                os.path.join(
                    "build",
                    "temp.%s-%d.%d"
                    % (
                        get_platform(),
                        sys.version_info[0],
                        sys.version_info[1],
                    ),
                ),
                "Release",
            )
            found = False
            for x in os.listdir(dlldir):
                if x.endswith(".lib") or x.endswith(".exp") or x.endswith(".dll"):
                    print(
                        "copying",
                        os.path.join(dlldir, x),
                        "-->",
                        os.path.join(self.build_lib, "awkward", x),
                    )
                    shutil.copyfile(
                        os.path.join(dlldir, x),
                        os.path.join(self.build_lib, "awkward", x),
                    )
                    found = True
            if not found:
                dlldir = os.path.join(dlldir, "Release")
                for x in os.listdir(dlldir):
                    if x.endswith(".lib") or x.endswith(".exp") or x.endswith(".dll"):
                        print(
                            "copying",
                            os.path.join(dlldir, x),
                            "-->",
                            os.path.join(self.build_lib, "awkward", x),
                        )
                        shutil.copyfile(
                            os.path.join(dlldir, x),
                            os.path.join(self.build_lib, "awkward", x),
                        )
                        found = True

            print("--- deleting libraries ----------------------------------------")
            for x in os.listdir(outerdir):
                if x.endswith(".pyd"):
                    print("deleting", os.path.join(outerdir, x))
                    os.remove(os.path.join(outerdir, x))

        print("--- begin normal install --------------------------------------")
        setuptools.command.install.install.run(self)

    def get_outputs(self):
        outerdir = os.path.join(
            os.path.join(
                "build",
                "lib.{}-{}.{}".format(
                    get_platform(),
                    sys.version_info[0],
                    sys.version_info[1],
                ),
            )
        )
        outputdir = os.path.join(outerdir, "awkward")
        outbase = self.install_lib.rstrip(os.path.sep)

        outputs = []

        for original in setuptools.command.install.install.get_outputs(self):
            if "egg-info" in original:
                outputs.append(original)
            if original.startswith(os.path.join(outbase, "awkward") + os.path.sep):
                outputs.append(original)

        for root, _, files in os.walk(outputdir):
            root = root[len(outerdir) :].lstrip(os.path.sep)
            for file in files:
                trial = os.path.join(outbase, os.path.join(root, file))
                if trial not in outputs:
                    outputs.append(trial)

        return outputs


setup(
    install_requires=install_requires,
    extras_require=extras,
    ext_modules=[CMakeExtension("awkward")],
    cmdclass={"build_ext": CMakeBuild, "install": Install, "build_py": BuildPy},
)

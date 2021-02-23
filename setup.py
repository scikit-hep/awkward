# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import platform
import subprocess
import sys
import multiprocessing
import shutil

import setuptools
import setuptools.command.build_ext
import setuptools.command.install
from setuptools import setup, Extension

# Note: never import from distutils before setuptools!
import distutils.util


try:
    import cmake

    CMAKE = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
except ImportError:
    CMAKE = "cmake"


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
            out = subprocess.check_output([CMAKE, "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(x.name for x in self.extensions)
            )

        for x in self.extensions:
            self.build_extension(x)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_args = []
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={0}".format(extdir),
            "-DPYTHON_EXECUTABLE={0}".format(sys.executable),
            "-DPYBUILD=ON",
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
            if (
                sys.maxsize > 2 ** 32
                and cmake_generator != "NMake Makefiles"
                and "Win64" not in cmake_generator
            ):
                cmake_args += ["-A", "x64"]

        elif "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j", str(multiprocessing.cpu_count())]

        if (
            platform.system() == "Darwin"
            and "MACOSX_DEPLOYMENT_TARGET" not in os.environ
        ):
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        build_dir = self.build_temp

        subprocess.check_call(
            [CMAKE, "-S", ext.sourcedir, "-B", build_dir] + cmake_args
        )
        subprocess.check_call([CMAKE, "--build", build_dir] + build_args)
        subprocess.check_call(
            [CMAKE, "--build", build_dir, "--config", cfg, "--target", "install"]
        )


def tree(x):
    print(x + (" (dir)" if os.path.isdir(x) else ""))
    if os.path.isdir(x):
        for y in os.listdir(x):
            tree(os.path.join(x, y))


if platform.system() == "Windows":

    class Install(setuptools.command.install.install):
        def run(self):
            outerdir = os.path.join(
                os.path.join(
                    "build",
                    "lib.%s-%d.%d"
                    % (
                        distutils.util.get_platform(),
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

            print("--- copying libraries -----------------------------------------")
            dlldir = os.path.join(
                os.path.join(
                    "build",
                    "temp.%s-%d.%d"
                    % (
                        distutils.util.get_platform(),
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
                    "lib.%s-%d.%d"
                    % (
                        distutils.util.get_platform(),
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

            for root, dirs, files in os.walk(outputdir):
                root = root[len(outerdir) :].lstrip(os.path.sep)
                for file in files:
                    trial = os.path.join(outbase, os.path.join(root, file))
                    if trial not in outputs:
                        outputs.append(trial)

            return outputs


else:

    class Install(setuptools.command.install.install):
        def run(self):
            outerdir = os.path.join(
                os.path.join(
                    "build",
                    "lib.%s-%d.%d"
                    % (
                        distutils.util.get_platform(),
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

            print("--- begin normal install --------------------------------------")
            setuptools.command.install.install.run(self)

        def get_outputs(self):
            outerdir = os.path.join(
                os.path.join(
                    "build",
                    "lib.%s-%d.%d"
                    % (
                        distutils.util.get_platform(),
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

            for root, dirs, files in os.walk(outputdir):
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
    cmdclass={"build_ext": CMakeBuild, "install": Install},
)

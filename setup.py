# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import os
import glob

from skbuild import setup
import setuptools

setup(name = "awkward1",
      packages = setuptools.find_packages(where="src"),
      package_dir = {"": "src"},
      data_files = ([
                     ("awkward1/include/awkward",             glob.glob("include/awkward/*.h")),
                     ("awkward1/include/awkward/cpu-kernels", glob.glob("include/awkward/cpu-kernels/*.h")),
                     ("awkward1/include/awkward/python",      glob.glob("include/awkward/python/*.h")),
                     ("awkward1/include/awkward/array",       glob.glob("include/awkward/array/*.h")),
                     ("awkward1/include/awkward/fillable",    glob.glob("include/awkward/fillable/*.h")),
                     ("awkward1/include/awkward/io",          glob.glob("include/awkward/io/*.h")),
                     ("awkward1/include/awkward/type",        glob.glob("include/awkward/type/*.h"))]),
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
      install_requires = open("requirements.txt").read().strip().split(),
      tests_require = open("requirements-test.txt").read().strip().split(),
      cmake_args=['-DBUILD_TESTING=OFF'],
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

#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
if sys.version_info <= (3, 0):
    sys.stdout.write("Sorry, Python 3.x is supported, not Python 2.x\n")
    sys.exit(1)

import argparse
import subprocess
import shutil
import os
import json
import glob
import multiprocessing

arguments = argparse.ArgumentParser()
arguments.add_argument("--clean", default=False, action="store_true")
arguments.add_argument("--release", action="store_true")
arguments.add_argument("--ctest", action="store_true")
arguments.add_argument("--no-buildpython", action="store_true")
arguments.add_argument("--no-dependencies", action="store_true")
arguments.add_argument("-j", default=str(multiprocessing.cpu_count()))
arguments.add_argument("--pytest", default=None)
args = arguments.parse_args()

args.buildpython = not args.no_buildpython
args.dependencies = not args.no_dependencies

try:
    git_config = open(".git/config").read()
except:
    git_config = ""

if "github.com/scikit-hep/awkward-1.0" not in git_config:
    arguments.error("localbuild must be executed in the head of the awkward-1.0 tree")

if args.clean:
    for x in ("localbuild", "awkward1", ".pytest_cache", "tests/__pycache__"):
        if os.path.exists(x):
            shutil.rmtree(x)
    sys.exit()

# Changes that would trigger a recompilation.
thisstate = {"release": args.release,
             "ctest": args.ctest,
             "buildpython": args.buildpython,
             "python_executable": sys.executable}

try:
    localbuild_time = os.stat("localbuild").st_mtime
except:
    localbuild_time = 0
try:
    laststate = json.load(open("localbuild/laststate.json"))
except:
    laststate = None

def check_call(args, env=None):
    print(" ".join(args))
    return subprocess.check_call(args, env=env)

# Refresh the directory if any configuration has changed.
if (os.stat("CMakeLists.txt").st_mtime >= localbuild_time or
    os.stat("localbuild.py").st_mtime >= localbuild_time or
    os.stat("setup.py").st_mtime >= localbuild_time or
    thisstate != laststate):

    if args.dependencies:
        check_call(["pip", "install", "-r", "requirements.txt", "-r", "requirements-test.txt"])

    if os.path.exists("localbuild"):
        shutil.rmtree("localbuild")

    newdir_args = ["-S", ".", "-Blocalbuild"]

    if args.release:
        newdir_args.append("-DCMAKE_BUILD_TYPE=Release")
    else:
        newdir_args.append("-DCMAKE_BUILD_TYPE=Debug")

    if args.ctest:
        newdir_args.append("-DBUILD_TESTING=ON")

    if args.buildpython:
        newdir_args.extend(["-DPYTHON_EXECUTABLE=" + thisstate["python_executable"], "-DPYBUILD=ON"])

    check_call(["cmake"] + newdir_args)
    json.dump(thisstate, open("localbuild/laststate.json", "w"))

# Build C++ normally; this might be a no-op if make/ninja determines that the build is up-to-date.
check_call(["cmake", "--build", "localbuild", "--", "-j" + args.j])

if args.ctest:
    check_call(["cmake", "--build", "localbuild", "--target", "test", "--", "CTEST_OUTPUT_ON_FAILURE=1", "--no-print-directory"])

# Build Python (copy sources to executable tree).
if args.buildpython:
    if os.path.exists("awkward1"):
        shutil.rmtree("awkward1")

    # Link (don't copy) the Python files into a built directory.
    for x in glob.glob("src/awkward1/**", recursive=True):
        olddir, oldfile = os.path.split(x)
        newdir  = olddir[3 + len(os.sep):]
        newfile = x[3 + len(os.sep):]
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        if not os.path.isdir(x):
            where = x
            for i in range(olddir.count(os.sep)):
                where = os.path.join("..", where)
            os.symlink(where, newfile)

    # The extension modules must be copied into the same directory.
    for x in glob.glob("localbuild/_ext*") + glob.glob("localbuild/libawkward*"):
        shutil.copyfile(x, os.path.join("awkward1", os.path.split(x)[1]))

    # localbuild must be in the library path for some operations.
    env = dict(os.environ)
    reminder = False
    if "awkward1" not in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = "awkward1:" + env.get("LD_LIBRARY_PATH", "")
        reminder = True

    # Run pytest on all or a subset of tests.
    if args.pytest is not None and not (os.path.exists(args.pytest) and not os.path.isdir(args.pytest) and not args.pytest.endswith(".py")):
        check_call(["python3", "-m", "pytest", "-vv", "-rs", args.pytest], env=env)

    # If you'll be using it interactively, you'll need awkward1 in the library path (for some operations).
    if reminder:
        print("")
        print("If you plan to use awkward1 outside of this tool, be sure to")
        print("")
        print("    export LD_LIBRARY_PATH=awkward1:$LD_LIBRARY_PATH")
        print("")

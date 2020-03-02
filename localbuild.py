#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import argparse
import subprocess
import shutil
import os
import sys
import json
import glob

arguments = argparse.ArgumentParser()
arguments.add_argument("--ctest", default=True, action="store_true")
arguments.add_argument("--no-ctest", action="store_true")
arguments.add_argument("--buildpython", default=True, action="store_true")
arguments.add_argument("--no-buildpython", action="store_true")
arguments.add_argument("--pytest", default=None)
args = arguments.parse_args()

if args.ctest == args.no_ctest:
    arguments.error("cannot pass --ctest and --no-ctest")

if args.buildpython == args.no_buildpython:
    arguments.error("cannot pass --buildpython and --no-buildpython")

args.ctest = not args.no_ctest
args.buildpython = not args.no_buildpython

thisstate = {"ctest": args.ctest, "buildpython": args.buildpython, "python_executable": sys.executable}

try:
    localbuild_time = os.stat("localbuild").st_mtime
except:
    localbuild_time = 0
try:
    laststate = json.load(open("localbuild/lastargs.json"))
except:
    laststate = None

# Refresh the directory if any configuration has changed.
if (os.stat("CMakeLists.txt").st_mtime >= localbuild_time or
    os.stat("localbuild.py").st_mtime >= localbuild_time or
    thisstate != laststate):

    subprocess.check_call(["pip", "install",
                           "-r", "requirements.txt",
                           "-r", "requirements-test.txt",
                           "-r", "requirements-docs.txt",
                           "-r", "requirements-dev.txt"])

    if os.path.exists("localbuild"):
        shutil.rmtree("localbuild")

    newdir_args = ["cmake", "-S", ".", "-B", "localbuild"]
    if args.ctest:
        newdir_args.append("-DBUILD_TESTING=ON")
    if args.buildpython:
        newdir_args.extend(["-DPYTHON_EXECUTABLE=" + thisstate["python_executable"], "-DPYBUILD=ON"])
    subprocess.check_call(newdir_args)
    json.dump(thisstate, open("localbuild/lastargs.json", "w"))

# Build C++ normally; this might be a no-op if make/ninja determines that the build is up-to-date.
subprocess.check_call(["cmake", "--build", "localbuild"])

if args.ctest:
    subprocess.check_call(["cmake", "--build", "localbuild", "--target", "test", "--", "CTEST_OUTPUT_ON_FAILURE=1"])

# Build Python (copy sources to executable tree).
if args.buildpython:
    if os.path.exists("awkward1"):
        shutil.rmtree("awkward1")

    # Maybe someday they can be symlinks.
    for x in glob.glob("src/awkward1/**", recursive=True):
        olddir, oldfile = os.path.split(x)
        newdir  = olddir[3 + len(os.sep):]
        newfile = x[3 + len(os.sep):]
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        if not os.path.isdir(x):
            # os.symlink(x, newfile)
            shutil.copyfile(x, newfile)

    # The extension modules must be copied over.
    for x in glob.glob("localbuild/layout*") + glob.glob("localbuild/types*") + glob.glob("localbuild/_io*"):
        shutil.copyfile(x, os.path.join("awkward1", os.path.split(x)[1]))

    # localbuild must be in the library path for some operations.
    env = dict(os.environ)
    reminder = False
    if "localbuild" not in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = "localbuild:" + env.get("LD_LIBRARY_PATH", "")
        reminder = True

    # for x in glob.glob("localbuild/lib*.so") + glob.glob("localbuild/lib*.a") + glob.glob("localbuild/lib*.dylib") + glob.glob("localbuild/*.lib") + glob.glob("localbuild/*.dll") + glob.glob("localbuild/*.exp"):
    #     shutil.copyfile(x, os.path.split(x)[1])

    # Run pytest on all or a subset of tests.
    if args.pytest is not None:
        subprocess.check_call(["python", "-m", "pytest", "-vv", "-rs", args.pytest], env=env)

    # If you'll be using it interactively, you'll need localbuild in the library path (for some operations).
    if reminder:
        print("")
        print("Remember to")
        print("")
        print("    export LD_LIBRARY_PATH=localbuild:$LD_LIBRARY_PATH")
        print("")

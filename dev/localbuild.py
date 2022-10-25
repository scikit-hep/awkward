import argparse
import itertools
import os
import pathlib
import shutil


def walk(directory):
    for x in os.listdir(directory):
        f = os.path.join(directory, x)
        yield f
        if os.path.isdir(f):
            yield from walk(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=pathlib.Path)
    args = parser.parse_args()

    if os.path.exists("awkward"):
        shutil.rmtree("awkward")

    # Link (don't copy) the Python files into a built directory.
    for x in walk(os.path.join("src", "awkward")):
        olddir, oldfile = os.path.split(x)
        newdir = olddir[3 + len(os.sep) :]
        newfile = x[3 + len(os.sep) :]
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        if not os.path.isdir(x):
            where = x
            for _ in range(olddir.count(os.sep)):
                where = os.path.join("..", where)
            os.symlink(where, newfile)

    # The extension modules must be copied into the same directory.
    for x in itertools.chain(args.dir.glob("_ext*"), args.dir.glob("libawkward*")):
        shutil.copyfile(x, os.path.join("awkward", os.path.split(x)[1]))

    # localbuild must be in the library path for some operations.
    env = dict(os.environ)
    reminder = False
    if "awkward" not in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = "awkward:" + env.get("LD_LIBRARY_PATH", "")
        reminder = True

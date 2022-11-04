"""Copy the header-only cpp headers into the various package directories that they are required"""
import pathlib
import shutil

root_path = pathlib.Path(__file__).absolute().parents[1]
source_path = root_path / "src" / "awkward" / "_connect" / "header-only"
dest_path = root_path / "awkward-cpp" / "header-only"

if __name__ == "__main__":
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(source_path, dest_path)

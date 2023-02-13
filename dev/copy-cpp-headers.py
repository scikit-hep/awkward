"""Copy the header-only cpp headers into the various package directories that they are required"""
import pathlib
import shutil

root_path = pathlib.Path(__file__).absolute().parents[1]
source_path = root_path / "header-only" / "include"
dest_paths = (
    root_path / "awkward-cpp" / "header-only",
    root_path / "src" / "awkward" / "_connect" / "header-only",
)

if __name__ == "__main__":
    for path in dest_paths:
        if path.exists():
            shutil.rmtree(path)
        shutil.copytree(source_path, path)

"""Copy the header-only cpp headers into the various package directories that they are required"""
import pathlib
import shutil

root_path = pathlib.Path(__file__).absolute().parents[1]
header_only_path = root_path / "header-only"
dest_paths = (
    root_path / "awkward-cpp" / "header-only",
    root_path / "src" / "awkward" / "_connect" / "header-only",
)


def copy_and_replace(source, dest):
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


if __name__ == "__main__":
    copy_and_replace(
        header_only_path / "include",
        root_path / "src" / "awkward" / "_connect" / "header-only",
    )
    copy_and_replace(
        header_only_path,
        root_path / "awkward-cpp" / "header-only",
    )

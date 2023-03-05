"""Copy the header-only cpp headers into the various package directories that they are required"""
import pathlib
import shutil

root_path = pathlib.Path(__file__).absolute().parents[1]


if __name__ == "__main__":
    header_only_path = root_path / "header-only"

    # Copy to Python package under `awkward`
    connect_path = (
        root_path / "src" / "awkward" / "_connect" / "header-only" / "awkward"
    )
    if connect_path.exists():
        shutil.rmtree(connect_path)
    connect_path.mkdir(parents=True)
    for path in header_only_path.rglob("*.h"):
        dest_path = connect_path / path.name
        shutil.copy(path, dest_path)

    # Copy to C++ package
    cpp_path = root_path / "awkward-cpp" / "header-only"
    if cpp_path.exists():
        shutil.rmtree(cpp_path)
    shutil.copytree(header_only_path, cpp_path)

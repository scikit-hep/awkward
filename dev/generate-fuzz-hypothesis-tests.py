import os
from hypothesis import given, settings, strategies as st
import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@given(
    st.lists(
        st.integers().filter(lambda x: x > 1), min_size=24, max_size=24, unique=True
    )
)
@settings(max_examples=1)
def genInput(x):
    input = [{"input": x}]
    with open(os.path.join(CURRENT_DIR, "..", "test_fuzz.yml"), "w") as fuzz:
        doc = yaml.dump(input, fuzz)
    genTest(str(x))


values = [
    "toindex = (ctypes.c_int64*len(toindex))(*toindex)",
    "frombitmask = [1, 1, 1, 1, 1]",
    "frombitmask = (ctypes.c_uint8*len(frombitmask))(*frombitmask)",
    "bitmasklength = 3",
    "validwhen = True",
    "lsb_order = True",
    "funcC = getattr(lib, 'awkward_BitMaskedArray_to_IndexedOptionArray64')",
    "funcC.restype = Error",
    "funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_bool, ctypes.c_bool)",
    "pytest_toindex = [0, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, 16, -1, -1, -1, -1, -1, -1, -1]",
    "assert toindex[:len(pytest_toindex)] == pytest.approx(pytest_toindex)",
]


def genTest(val):
    num = 0
    with open(
        os.path.join(
            CURRENT_DIR,
            "..",
            "tests-spec",
            "test_fuzz_pyawkward_BitMaskedArray_to_IndexedOptionArray64.py",
        ),
        "w",
    ) as f:
        f.write("import ctypes\nimport pytest\nfrom __init__ import lib, Error\n\n")
        f.write("def test_cpuawkward_BitMaskedArray_to_IndexedOptionArray64():\n")
        f.write("\t" + "toindex=" + val + "\n")
        for i in range(11):
            f.write("\t" + values[num] + "\n")
            num += 1


if __name__ == "__main__":
    genInput()

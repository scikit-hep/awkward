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
def gen_input(x):
    doc = None
    with open(
        os.path.join(
            CURRENT_DIR,
            "..",
            "fuzz_test.yml",
        ),
        "r",
    ) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
        doc[0]["inputs"][0]["toindex"] = x
    with open(
        os.path.join(
            CURRENT_DIR,
            "..",
            "fuzz_test.yml",
        ),
        "w",
    ) as y:
        yaml.dump(doc, y, default_flow_style=True)
        gen_test(doc)


def gen_test(doc):
    index = 0
    file_name = "test_fuzz_py" + doc[index]["name"] + ".py"
    with open(os.path.join(CURRENT_DIR, "..", "tests-spec", file_name), "w") as f:
        for name in doc:
            num = 1
            f.write("import ctypes\nimport pytest\nfrom __init__ import lib, Error\n\n")
            f.write("def " + str(name["name"]) + "_" + str(num) + "():\n")
            for arg in name["args"]:
                f.write(
                    "\t"
                    + str(arg["name"])
                    + " = "
                    + str(name["inputs"][num - 1][str(arg["name"])])
                    + "\n"
                )
                num += 1
            f.write("\t" + "funcPy = getattr(kernels," + str(name["name"]) + ")\n")
            f.write("\tfuncPy(")
            num = 0
            for arg in name["args"]:
                if num != 0:
                    f.write(",")
                f.write(str(arg["name"]) + "=" + str(arg["name"]))
                num += 1
            f.write(")\n")


if __name__ == "__main__":
    gen_input()

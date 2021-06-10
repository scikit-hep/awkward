import os
from hypothesis import strategies as st
from hypothesis.strategies import composite
import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

@composite
def generate(draw, elements=st.integers()):
    #size=draw(st.integers().filter(lambda x: x > 1))
    content = draw(st.lists(elements,min_size=5, max_size=5, unique=True))
    mask = draw(st.lists(elements,min_size=5, max_size=5, unique=True))
    length=draw(st.integers(min_value=0,max_value=5))
    valid_when=draw(st.booleans())
    lsb_order=draw(st.booleans())
    return (mask,content,valid_when,length,lsb_order)


def gen_input():
    ex=generate().example()
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
        doc[0]["inputs"][0]["toindex"] = ex[0]
        doc[0]["inputs"][0]["frombitmask"] = ex[1]
        doc[0]["inputs"][0]["bitmasklength"] = ex[2]
        doc[0]["inputs"][0]["validwhen"] = ex[3]
        doc[0]["inputs"][0]["lsb_order"] = ex[4]
    with open(
        os.path.join(
            CURRENT_DIR,
            "..",
            "fuzz_test.yml",
        ),
        "w",
    ) as y:
        yaml.dump(doc, y, default_flow_style=True)
        gen_pytest(doc)
        gen_ctest(doc)


def gen_pytest(doc):
    index = 0
    file_name = "test_fuzz_py" + doc[index]["name"] + ".py"
    with open(os.path.join(CURRENT_DIR, "..", "tests-spec", file_name), "w") as f:
        for name in doc:
            num = 1
            f.write("import pytest\nimport kernels\n\n")
            f.write("def " +"test_"+ str(name["name"]) + "_" + str(num) + "():\n")
            for arg in name["args"]:
                f.write(
                    "\t"
                    + str(arg["name"])
                    + " = "
                    + str(name["inputs"][num - 1][str(arg["name"])])
                    + "\n"
                )
                num += 1
            f.write("\t" + "funcPy = getattr(kernels," + "'"+ str(name["name"]) + "'"+ ")\n")
            f.write("\tfuncPy(")
            num = 0
            for arg in name["args"]:
                if num != 0:
                    f.write(",")
                f.write(str(arg["name"]) + "=" + str(arg["name"]))
                num += 1
            f.write(")\n")


def gen_ctest(doc):
    index = 0
    file_name = "test_fuzz_py" + doc[index]["name"] +"64"+ ".py"
    with open(os.path.join(CURRENT_DIR, "..", "tests-cpu-kernels", file_name), "w") as f:
        for name in doc:
            num = 1
            f.write("import ctypes\nimport pytest\nfrom __init__ import lib, Error\n\n")
            f.write("def " +"test_"+ str(name["name"])+ "64" + "_" + str(num) + "():\n")
            for arg in name["args"]:
                f.write(
                    "\t"
                    + str(arg["name"])
                    + " = "
                    + str(name["inputs"][num - 1][str(arg["name"])])
                    + "\n"
                )
                if(num==1):
                    f.write("\t"+str(arg["name"])+"="+"(ctypes.c_int64*len("+str(arg["name"])+"))(*"+str(arg["name"])+")\n")
                if(num==2):
                    f.write("\t"+str(arg["name"])+"="+"(ctypes.c_unit8*len("+str(arg["name"])+"))(*"+str(arg["name"])+")\n")
                num += 1
            f.write("\t" + "funcC = getattr(lib," + "'"+ str(name["name"]) + "'"+ ")\n")
            f.write("\tfuncC(")
            num = 0
            for arg in name["args"]:
                if num != 0:
                    f.write(",")
                f.write(str(arg["name"]) + "=" + str(arg["name"]))
                num += 1
            f.write(")\n")


if __name__ == "__main__":
    gen_input()

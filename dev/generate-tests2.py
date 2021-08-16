import os
import yaml
import datetime
import shutil

import yaml
from numpy import uint8

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

result = []


class Argument(object):
    __slots__ = ("name", "typename", "direction", "role")

    def __init__(self, name, typename, direction, role="default"):
        self.name = name
        self.typename = typename
        self.direction = direction
        self.role = role


def genpykernels():
    print("Generating Python kernels")
    prefix = """
from numpy import uint8
kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1
"""

    new_file = os.path.join(CURRENT_DIR, "..", "new_tests")
    if os.path.exists(new_file):
        shutil.rmtree(new_file)
    os.mkdir(new_file)
    with open(os.path.join(new_file, "__init__.py"), "w") as f:
        f.write(
            """# AUTO GENERATED ON {0}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

""".format(
                datetime.datetime.now().isoformat().replace("T", " AT ")[:22]
            )
        )

    with open(
        os.path.join(CURRENT_DIR, "..", "new_tests", "kernels.py"), "w"
    ) as outfile:
        outfile.write(prefix)
        with open(
            os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")
        ) as specfile:
            indspec = yaml.safe_load(specfile)["kernels"]
            for spec in indspec:
                if "def " in spec["definition"]:
                    outfile.write(spec["definition"] + "\n")
                    for childfunc in spec["specializations"]:
                        outfile.write(
                            "{0} = {1}\n".format(childfunc["name"], spec["name"])
                        )
                    outfile.write("\n\n")


class Data:
    def __init__(self, func, args):
        self.function_name = func
        self.args = args

    def returnValues(self, type, value):
        # print("Type= "+str(type)+" Value= "+str(value))
        if "List" in type:
            arr = []
            for item in value.split(","):
                arr.append(int(item))
            return arr
        elif type == "bool":
            return True if value == "1" else False
        else:
            return int(value)

    def printTest(self, arguments):
        str = ""
        ind = -1
        temp = {}
        for items in self.args:
            ind += 1
            # str+=arguments[ind]['name']+'='+self.returnValues(arguments[ind]['type'],items)+'\n'
            temp["Function_Name"] = self.function_name
            temp[arguments[ind]["name"]] = self.returnValues(
                arguments[ind]["type"], items
            )
        result.append(temp)


def remove_duplicates():
    print("Removing Duplicates from Output File")
    file = open(os.path.join(CURRENT_DIR, "..", "outputs.txt"), "r")
    lines = file.readlines()
    unique = set()
    for line in lines:
        if line[0:8] == "awkward_":
            unique.add(line)
    file.close()
    file = open(os.path.join(CURRENT_DIR, "..", "outputs.txt"), "w").writelines(unique)


def generateTests():
    specifications = open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification.yml"), "r"
    )
    dict = yaml.safe_load(specifications)["kernels"]
    lines = open(os.path.join(CURRENT_DIR, "..", "outputs.txt"), "r").readlines()
    for line in lines:
        data = Data(line.split(":")[0], line.split(":")[1 : len(line.split(":"))])
        function_name = data.function_name
        for kernel in dict:
            if function_name == kernel["name"]:
                data.printTest(kernel["specializations"][0]["args"])


def sanitizeLine(line):
    arr = []
    args = line.split(":")[1 : len(line.split(":"))]
    for arg in args:
        temp = ""
        for ch in arg:
            if ch.isdigit() or ch == "," or ch == ":":
                temp += ch
        if temp != "":
            if temp[len(temp) - 1] == "," or temp[len(temp) - 1] == ":":
                temp = temp[0 : len(temp) - 1]
            arr.append(temp)
    return arr


def generateYaml():
    print("Generating YAML File")
    line_num = 0
    specifications = open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification.yml"), "r"
    )
    dict = yaml.safe_load(specifications)["kernels"]
    lines = open(os.path.join(CURRENT_DIR, "..", "outputs.txt"), "r").readlines()
    for line in lines:
        data = Data(line.split(":")[0], sanitizeLine(line))
        line_num += 1
        # print("Line Number: "+str(line_num)+" Line= "+line+"the Last char: "+str(line[len(line)-1]))
        function_name = data.function_name
        for kernel in dict:
            if function_name == kernel["name"]:
                if kernel["automatic-tests"] == False:
                    break
                data.printTest(kernel["specializations"][0]["args"])
    x = {"Data": result}
    with open("generated-data.yml", "w") as ymlfile:
        yaml.dump(x, ymlfile, default_flow_style=None)


def generateTests():
    print("Generating Python Tests")
    specifications = open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification.yml"), "r"
    )
    dict = yaml.safe_load(specifications)["kernels"]
    datayml = open(os.path.join(CURRENT_DIR, "..", "generated-data.yml"), "r")
    data = yaml.safe_load(datayml)["Data"]
    num = -1
    for test in data:
        num += 1
        func = "test_" + test["Function_Name"] + "_" + str(num) + ".py"
        with open(os.path.join(CURRENT_DIR, "..", "new_tests", func), "w") as file:
            file.write("import kernels\n\n")
            funcName = "def test_" + test["Function_Name"] + "():\n"
            file.write(funcName)
            for kernel in dict:
                if kernel["name"] == test["Function_Name"]:
                    args = kernel["specializations"][0]["args"]
                    if (len(test) - 1) == len(args):
                        output_line = ""
                        outputs = []
                        for arg in args:
                            line = ""
                            if arg["dir"] == "out":
                                outputs.append(arg["name"])
                                line = (
                                    "\t"
                                    + arg["name"]
                                    + "="
                                    + str([123] * len(test[arg["name"]]))
                                    + "\n"
                                )
                                output_line += (
                                    "\tpytest_"
                                    + arg["name"]
                                    + "="
                                    + str(test[arg["name"]])
                                    + "\n"
                                )
                            else:
                                line = (
                                    "\t"
                                    + arg["name"]
                                    + "="
                                    + str(test[arg["name"]])
                                    + "\n"
                                )
                            file.write(line)
                        file.write("\tfuncPy = getattr(kernels, ")
                        line = "'" + test["Function_Name"] + "'"
                        file.write(line + ")\n")
                        file.write("\tfuncPy(")
                        line = ""
                        for arg in args:
                            line += str(arg["name"]) + "=" + str(arg["name"]) + ","
                        file.write(line[0 : len(line) - 1])
                        file.write(")\n")
                        file.write(output_line)
                        for output in outputs:
                            file.write(
                                "\tassert "
                                + str(output)
                                + "== pytest_"
                                + str(output)
                                + "\n"
                            )
                    else:
                        file.write('\tprint("ExceptionOccured")')


if __name__ == "__main__":
    genpykernels()
    remove_duplicates()
    generateYaml()
    generateTests()

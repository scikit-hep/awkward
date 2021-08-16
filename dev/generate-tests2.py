import os
import yaml
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

result = []


class Data:
    def __init__(self, func, args):
        self.function_name = func
        self.args = args

    def returnValues(self, type, value):
        str = ""
        if "List" in type:
            str += "["
            for item in value.split(","):
                str += item + ","
            str = str[0 : len(str) - 1] + "]"
            return str
        else:
            return value

    def printTest(self, arguments):
        str = ""
        ind = -1
        temp = {}
        for items in self.args:
            ind += 1
            # str+=arguments[ind]['name']+'='+self.returnValues(arguments[ind]['type'],items)+'\n'
            temp["Function_Name"] = self.function_name
            temp["{}".format(arguments[ind]["name"])] = self.returnValues(
                arguments[ind]["type"], items
            )
        result.append(temp)


def remove_duplicates():
    file = open(os.path.join(CURRENT_DIR, "..", "outputs.txt"), "r")
    lines = file.readlines()
    unique = set()
    for line in lines:
        if line[0:8] == "awkward_":
            unique.add(line)
    file.close()
    file = open(os.path.join(CURRENT_DIR, "..", "outputs.txt"), "w").writelines(unique)


def test():
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


def generateYaml():
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
    x = {"Data": result}
    with open("generated-data.yml", "w") as ymlfile:
        yaml.dump(x, ymlfile, default_flow_style=False)


if __name__ == "__main__":
    remove_duplicates()
    generateYaml()

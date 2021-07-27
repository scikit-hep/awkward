import re
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
num = 0


def get_line(args):
    new = remove_comments(args)
    str = "".join(new.splitlines()).replace(" ", "")
    start = str.find("(")
    stop = len(str) - 1
    temp = str[start:stop]
    arguments = temp[temp.find(",") + 1 : len(temp) - 1]
    values = arguments.split(",")
    function_name = str[str.find(":") + 2 : start]
    if function_name[len(function_name) - 1] == ">":
        function_name = function_name[0 : function_name.find("<")]
    line = 'std::cout<<"awkward_' + function_name + '"<<'
    for value in values:
        line += '"' + value + '="<<' + value + "<<"
    line = line[0 : len(line) - 2]
    line = line + ";"
    return line


def remove_comments(args):
    s = """"""
    for ln in args.splitlines():
        z = ""
        f = 0
        for ch in ln:
            if ch == "/":
                f = 1
            elif f == 0:
                z += ch
        s += z
    return s


for root, subdirs, files in os.walk(os.path.join(CURRENT_DIR, "src", "libawkward")):
    record = False
    block = """"""
    new_file = """"""
    for file in files:
        new_file = """"""
        x = open(os.path.join(root, file), "r")
        lines = x.readlines()
        for line in lines:
            if re.search(r"struct\sError\serr\s=\skernel::\w+", line):
                block += line
                record = True
            elif re.search(r"[)];", line) and record == True:
                block += line
                stdout = get_line(block)
                block = """"""
                record = False
                new_file += line
                new_file += stdout + "\n"
                num += 1
                continue
            elif record == True:
                block += line
            new_file += line
        with open(os.path.join(root, file), "w") as f:
            f.write(new_file)

print("Added cout in " + str(num) + " places")

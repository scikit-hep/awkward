import re
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
num = 0


def get_line(args):
    str = "".join(args.splitlines()).replace(" ", "")
    start = str.find("(")
    stop = len(str) - 1
    temp = str[start:stop]
    arguments = temp[temp.find(",") + 1 : len(temp) - 1]
    values = arguments.split(",")
    function_name = str[str.find(":") + 2 : start]
    if function_name[len(function_name) - 1] == ">":
        function_name = function_name[0 : function_name.find("<")]
    line = 'cout<<"awkward_' + function_name + '"<<'
    for value in values:
        line += '"' + value + '="<<' + value + "<<"
    line = line[0 : len(line) - 2]
    line = "".join(ch for ch in line if not ch.isupper())
    line = "".join(ch for ch in line if ch != "/")
    return line


for root, subdirs, files in os.walk(os.path.join(CURRENT_DIR, "src", "libawkward")):
    record = False
    block = """"""
    new_file = """"""
    for file in files:
        # print(os.path.join(root,file))
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
                new_file += stdout
                num += 1
            elif record == True:
                block += line
            new_file += line
        with open(os.path.join(root, file), "w") as f:
            f.write(new_file)

print("Added cout in " + str(num) + " places")

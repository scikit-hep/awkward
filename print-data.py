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


def reducer():
    new_file = """"""
    global num
    record = False
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Reducer.cpp"), "r")
    lines = x.readlines()
    for line in lines:
        if re.search(r"struct\sError\serr\s=\skernel::\w+", line):
            record = True
        elif re.search(r"[)];", line) and record == True:
            record = False
            new_file += line
            new_file += """
    std::cout<<"reduce_argmin_64:";
    std::cout<<"parents=";
    for(int i=0;i<parents.length();i++){
        std::cout<<data[i]<<",";
    }
    std::cout<<"lenparents="<<parents.length()<<"outlength="<<outlength<<std::endl;
            """
            num += 1
            continue
        new_file += line
    x.close()
    return new_file


def slicer():
    new_file = """"""
    global num
    record = False
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Slice.cpp"), "r")
    lines = x.readlines()
    for line in lines:
        if re.search(r"struct\sError\serr\s=\skernel::\w+", line):
            record = True
        elif re.search(r"[)];", line) and record == True:
            record = False
            new_file += line
            new_file += 'std::cout<<"awkward_slicearray_ravel_64:index=";index.printMe();std::cout<<";index_=";index_.printMe();std::cout<<";ndim_="<<ndim()<<";shape=";for (auto x : shape_)std::cout << x << ",";std::cout<<"strides=";for (auto x : strides_)std::cout << x << ",";std::cout<<std::endl;'
            num += 1
            continue
        new_file += line
    x.close()
    return new_file


def others(path):
    new_file = """"""
    global num
    x = open(path, "r")
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
    return new_file


for root, subdirs, files in os.walk(os.path.join(CURRENT_DIR, "src", "libawkward")):
    record = False
    block = """"""
    new_file = """"""
    for file in files:
        new_file = """"""
        if file == "Reducer.cpp":
            temp = reducer()
            with open(os.path.join(root, file), "w") as f:
                f.write(temp)
                f.close()
        if file == "Slice.cpp":
            temp = slicer()
            with open(os.path.join(root, file), "w") as f:
                f.write(temp)
                f.close()

print("Added cout in " + str(num) + " places")

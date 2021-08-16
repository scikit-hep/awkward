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
    line = 'std::cout<<"awkward_' + function_name + ":" + '";'
    # if function_name=='NumpyArray_getitem_next_null_64': #Exception Case
    #     return ''
    for value in values:
        # if "reinterpret_cast" in value:
        #     value = value[value.find(">") + 2 : len(value) - 1]
        if "tocarryraw" in value:
            line += """
            for(int i=0;i<length();i++){
        std::cout<<tocarryraw.data()[i]<<",";
    }
    std::cout<<":";\n"""
        elif "lencontents" in value:
            line += """
            for(int i=0;i<length();i++){
        std::cout<<lencontents.data()[i]<<",";
    }
    std::cout<<":";\n"""
        elif "offsetsraws" in value:
            line += """
            for(int i=0;i<length();i++){
        std::cout<<offsetsraws.data()[i]<<",";
    }
    std::cout<<":";\n"""
        elif "reinterpret_cast" in value:
            line += (
                "for(int i=0;i<this->length();i++){std::cout <<"
                + "reinterpret_cast<int64_t*>"
                + value[value.find(">") + 1 : len(value)]
                + '[i]<<",";}'
            )
        elif "data()" in value and "reinterpret_cast" not in value:
            line += value.replace("data()", "printMe()") + ";" + 'std::cout<<":";'
        elif value.isdigit():
            line += "std::cout<<" + '"' + value + '"<<":";'
        elif "kernel::" in value:
            continue
        else:
            line += "std::cout<<" + value + '<<":";'
    line = line[0 : len(line)] + "std::cout<<std::endl;"
    line = line.replace("&", "")
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
    std::cout<<"awkward_reduce_argmin_64:";
    for(int i=0;i<parents.length();i++){
        std::cout<<data[i]<<",";
    }
    std::cout<<":";
    std::cout<<parents.length()<<":"<<outlength<<std::endl;
            """
            num += 1
            continue
        new_file += line
    x.close()
    return new_file


def insertPrintMe():
    new_file = """"""
    global num
    flag = 0
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Index.cpp"), "r")
    lines = x.readlines()
    for line in lines:
        new_file += line
        if "template class" in line and flag == 0:
            new_file += """  template <typename T>
    void
    IndexOf<T>::printMe() const{
        for (int64_t i = 0;  i < length();  i++) {
        std::cout << data()[i] << ", ";
        }
    }"""
            flag = 1
    x.close()
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Index.cpp"), "w")
    x.write(new_file)
    x.close()
    flag = 0
    new_file = """"""
    x = open(
        os.path.join(CURRENT_DIR, "src", "libawkward", "array", "NumpyArray.cpp"), "r"
    )
    lines = x.readlines()
    for line in lines:
        new_file += line
        if "NumpyArray::classname" in line and flag == 0:
            flag = 1
        if "}" in line and flag == 1:
            new_file += """void NumpyArray::printMe() const{
    for (int64_t i = 0;  i < length();  i++) {
        std::cout << (reinterpret_cast<int64_t*>(data()))[i];
      }
  }"""
            flag = 2
    x.close()
    x = open(
        os.path.join(CURRENT_DIR, "src", "libawkward", "array", "NumpyArray.cpp"), "w"
    )
    x.write(new_file)
    x.close()


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
            new_file += 'std::cout<<"awkward_slicearray_ravel_64:";index.printMe();std::cout<<":";index_.printMe();std::cout<<":"<<ndim()<<":";for (auto x : shape_)std::cout << x << ",";std::cout<<":";for (auto x : strides_)std::cout << x << ",";std::cout<<std::endl;\n'
            num += 1
            continue
        new_file += line
    x.close()
    return new_file


def identities():
    new_file = """"""
    global num
    record = False
    flag = 0
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Identities.cpp"), "r")
    lines = x.readlines()
    for line in lines:
        if re.search(r"struct\sError\serr\s=\skernel::\w+", line):
            record = True
        elif re.search(r"[)];", line) and record == True:
            flag += 1
            record = False
            new_file += line
            new_file += 'std::cout<<"awkward_Identities_getitem_carry_64:";printMe();std::cout<<":";carry.printMe();std::cout<<":";std::cout<<carry.length()<<":"<<width_<<":"<<length_<<std::endl;\n'
            num += 1
            continue
        if flag == 2 and "return out;" in line:
            flag = 3
        if flag == 3 and "}" in line:
            flag = 4
            new_file += line
            new_file += """ template <typename T>
  void
  IdentitiesOf<T>::printMe() const{
   for (int64_t i = 0; i < length(); i++) {
    std::cout << data()[i] << ", ";
   }
  }"""
            continue
        new_file += line
    x.close()
    return new_file


def content():
    new_file = """"""
    global num
    block = """"""
    record = False
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Content.cpp"), "r")
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


def index():
    new_file = """"""
    global num
    record = False
    x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "Index.cpp"), "r")
    lines = x.readlines()
    function_name = ""
    for line in lines:
        if re.search(r"struct\sError\serr\s=\skernel::\w+", line):
            record = True
            function_name = line[line.find(":") + 2 : len(line) - 2]
            if function_name[len(function_name) - 1] == ">":
                function_name = function_name[0 : function_name.find("<")]
        elif re.search(r"[)];", line) and record == True:
            record = False
            new_file += line
            if function_name == "Index_iscontiguous":
                new_file += '\tstd::cout<<"New Line";std::cout<<std::endl;std::cout<<"awkward_Index_iscontiguous:";std::cout<<result;std::cout<<":";printMe();std::cout<<":"<<length_<<std::endl;'
            else:
                new_file += '\tstd::cout<<"New Line";std::cout<<std::endl;std::cout<<"awkward_Index_to_Index64:"<<ptr_.get()[(size_t)offset_];std::cout<<":";printMe();std::cout<<":"<<length_<<std::endl;\n'
            num += 1
            continue
        new_file += line
    x.close()
    return new_file


def union():
    new_file = """"""
    global num
    record = False
    x = open(
        os.path.join(
            CURRENT_DIR, "src", "libawkward", "layoutbuilder", "UnionArrayBuilder.cpp"
        ),
        "r",
    )
    lines = x.readlines()
    for line in lines:
        if re.search(r"struct\sError\serr\s=\skernel::\w+", line):
            record = True
        elif re.search(r"[)];", line) and record == True:
            record = False
            new_file += line
            new_file += '\tstd::cout<<"awkward_UnionArray_regular_index:"<<lentags<<":";tags.printMe();std::cout<<":"<<lentags<<std::endl;\n'
            num += 1
            continue
        new_file += line
    x.close()
    return new_file


def awkward():
    for root, subdirs, files in os.walk(
        os.path.join(CURRENT_DIR, "src", "libawkward", "array")
    ):
        for file in files:
            new_file = """"""
            global num
            record = False
            block = """"""
            x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "array", file), "r")
            lines = x.readlines()
            for line in lines:
                if re.search(r"struct\sError\s(err|err1|err2)\s=", line):
                    record = True
                    block += line
                elif re.search(r"[)];", line) and record == True:
                    record = False
                    new_file += line
                    block += line
                    stdout = get_line(block)
                    block = """"""
                    new_file += stdout + "\n"
                    num += 1
                    continue
                elif record == True:
                    block += line
                new_file += line
            x.close()
            x = open(os.path.join(CURRENT_DIR, "src", "libawkward", "array", file), "w")
            x.write(new_file)
            x.close()


for root, subdirs, files in os.walk(os.path.join(CURRENT_DIR, "src", "libawkward")):
    record = False
    block = """"""
    new_file = """"""
    for file in files:
        new_file = """"""
        # if file == "Reducer.cpp":
        #     temp = reducer()
        #     with open(os.path.join(root, file), "w") as f:
        #         f.write(temp)
        #         f.close()
        # if file == "Slice.cpp":
        #     insertPrintMe()
        #     temp = slicer()
        #     with open(os.path.join(root, file), "w") as f:
        #         f.write(temp)
        #         f.close()
        # if file == "Identities.cpp":
        #     temp = identities()
        #     with open(os.path.join(root, file), "w") as f:
        #         f.write(temp)
        #         f.close()
        # if file == "Content.cpp":
        #     temp = content()
        #     with open(os.path.join(root, file), "w") as f:
        #         f.write(temp)
        #         f.close()
        if file == "Index.cpp":
            temp = index()
            with open(os.path.join(root, file), "w") as f:
                f.write(temp)
                f.close()
        # if file == "UnionArrayBuilder.cpp":
        #     temp = union()
        #     with open(os.path.join(root, file), "w") as f:
        #         f.write(temp)
        #         f.close()

insertPrintMe()
# awkward()
print("Added cout in " + str(num) + " places")

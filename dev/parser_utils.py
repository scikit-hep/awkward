# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE


def indent_code(code, indent):
    finalcode = ""
    for line in code.splitlines():
        finalcode += " " * indent + line + "\n"
    return finalcode

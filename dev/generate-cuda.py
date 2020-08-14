# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import argparse
import ast
import os
from collections import OrderedDict

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
KERNEL_WHITELIST = [
    "awkward_new_Identities",
    "awkward_RegularArray_num",
    "awkward_IndexedArray_overlay_mask",
    "awkward_IndexedArray_mask",
    "awkward_zero_mask",
]


def traverse(node, args={}):
    if node.__class__.__name__ == "For":
        code = "if (thread_id < length) {\n"
        for subnode in node.body:
            code += traverse(subnode, args)
        code += "}\n"
    elif node.__class__.__name__ == "Name":
        code = node.id
    elif node.__class__.__name__ == "BinOp":
        code = "{0} {1} {2}".format(
            traverse(node.left), traverse(node.op), traverse(node.right),
        )
    elif node.__class__.__name__ == "UnaryOp":
        if node.op.__class__.__name__ == "USub":
            code = "-{0}".format(node.operand.value)
    elif node.__class__.__name__ == "Sub":
        code = "-"
    elif node.__class__.__name__ == "Subscript":
        if node.slice.value.__class__.__name__ == "Name" and node.slice.value.id == "i":
            code = node.value.id + "[thread_id]"
        elif node.slice.value.__class__.__name__ == "Name":
            code = node.value.id + "[" + node.slice.value.id + "]"
        else:
            code = traverse(node.slice.value)
    elif node.__class__.__name__ == "Call":
        assert len(node.args) == 1
        code = "({0})({1})".format(node.func.id, traverse(node.args[0]))
    elif node.__class__.__name__ == "Constant":
        code = node.value
    elif node.__class__.__name__ == "Assign":
        assert len(node.targets) == 1
        left = traverse(node.targets[0], args)
        if "[" in left:
            left = left[: left.find("[")]
        code = ""
        if left not in args.keys() and ("*" + left) not in args.keys():
            flag = True
        else:
            flag = False
        if node.value.__class__.__name__ == "Name" and node.value.id == "i":
            code = ""
            if flag:
                code += "auto "
            code += "{0} = thread_id;\n".format(traverse(node.targets[0]))
        else:
            if node.value.__class__.__name__ == "IfExp":
                if flag:
                    code = "if ({0}) {{\n auto {1} = {2};\n }} else {{\n auto {1} = {3};\n }}\n".format(
                        node.value.test.id,
                        traverse(node.targets[0]),
                        traverse(node.value.body),
                        traverse(node.value.orelse),
                    )
                else:
                    code = "if ({0}) {{\n {1} = {2};\n }} else {{\n {1} = {3};\n }}\n".format(
                        node.value.test.id,
                        traverse(node.targets[0]),
                        traverse(node.value.body),
                        traverse(node.value.orelse),
                    )
            elif node.value.__class__.__name__ == "Compare":
                code = ""
                if flag:
                    code += "auto "
                if (
                    len(node.value.ops) == 1
                    and node.value.ops[0].__class__.__name__ == "Lt"
                ):
                    code += "{0} = {1} < {2};\n".format(
                        traverse(node.targets[0]),
                        traverse(node.value.left),
                        traverse(node.value.comparators[0]),
                    )
            else:
                code = ""
                if flag:
                    code += "auto "
                code += "{0} = {1};\n".format(
                    traverse(node.targets[0]), traverse(node.value)
                )
    else:
        raise Exception("Unhandled node {0}".format(node.__class__.__name__))
    return code


def getbody(pycode, args):
    code = ""
    tree = ast.parse(pycode).body[0]
    for node in tree.body:
        code += traverse(node, args)
    return code


def getcudaname(name):
    assert name.startswith("awkward_")
    assert name.startswith("awkwardcuda_") is False
    return name[: len("awkward")] + "cuda" + name[len("awkward") :]


def getctype(typename):
    pointercount = 0
    while "List[" in typename:
        typename = typename[5:]
        typename = typename[:-1]
        pointercount += 1
    cpptype = typename + "*" * pointercount
    return cpptype


def gettemplateargs(spec):
    templateargs = {}
    if "specializations" in spec.keys():
        typelist = []
        count = 0
        templascii = 65
        for childfunc in spec["specializations"]:
            for i in range(len(childfunc["args"])):
                if len(typelist) < i + 1:
                    typelist.append(list(childfunc["args"][i].values())[0])
                else:
                    if typelist[i] != list(childfunc["args"][i].values())[0]:
                        templateargs[list(childfunc["args"][i].keys())[0]] = chr(
                            templascii
                        )
                        count += 1
                        templascii += 1
    return templateargs


def getparentargs(templateargs, spec):
    args = OrderedDict()
    if "specializations" in spec.keys():
        for arg in spec["specializations"][0]["args"]:
            argname = list(arg.keys())[0]
            if list(arg.keys())[0] in templateargs.keys():
                if "*" in getctype(list(arg.values())[0]):
                    argname = "*" + argname
                args[argname] = templateargs[list(arg.keys())[0]]
            else:
                args[argname] = getctype(list(arg.values())[0])
    else:
        for arg in spec["args"]:
            argname = list(arg.keys())[0]
            if argname in spec["outparams"]:
                argname = "*" + argname
            if "*" not in argname:
                args[argname] = list(arg.values())[0]
    return args


def getchildargs(childfunc, spec):
    args = OrderedDict()
    for arg in childfunc["args"]:
        for argname, typename in arg.items():
            args[argname] = getctype(typename)
    return args


def gettemplatestring(templateargs):
    count = 0
    templatestring = ""
    for x in templateargs.values():
        if count == 0:
            templatestring += "typename " + x
            count += 1
        else:
            templatestring += ", typename " + x
    return templatestring


def getdecl(name, args, templatestring, parent=False, solo=False):
    code = ""
    if templatestring != "":
        code += "<" + templatestring + ">\n"
    if parent:
        code += "__global__\n"
    count = 0
    for key, value in args.items():
        if count == 0:
            params = value + " " + key
            count += 1
        else:
            params += ", " + value + " " + key
    if parent:
        code += "void cuda" + name[len("awkward") :] + "(" + params + ") {\n"
    else:
        code += "ERROR " + getcudaname(name) + "(" + params + ") {\n"
    return code


def gettemplatetypes(spec, templateargs):
    count = 0
    code = ""
    for arg in spec["args"]:
        for argname, typename in arg.items():
            if argname in templateargs.keys():
                if count == 0:
                    code += getctype(typename).replace("*", "")
                    count += 1
                else:
                    code += ", " + getctype(typename).replace("*", "")
    return code


def getparamnames(args):
    count = 0
    for arg in args.keys():
        if count == 0:
            code = arg
            count += 1
        else:
            code += ", " + arg
    return code


def getcode(indspec):
    templateargs = gettemplateargs(indspec)
    args = getparentargs(templateargs, indspec)
    if "specializations" in indspec.keys():
        templatestring = gettemplatestring(templateargs)
    else:
        templatestring = ""
        args = getchildargs(indspec, indspec)
    code = getdecl(
        indspec["name"],
        args,
        templatestring,
        parent=True,
        solo="specializations" in indspec.keys(),
    )
    code += """  int64_t block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int64_t thread_id = block_id * blockDim.x + threadIdx.x;
"""
    code += getbody(indspec["definition"], args)
    code += "}\n\n"
    if "specializations" in indspec.keys():
        for childfunc in indspec["specializations"]:
            args = getchildargs(childfunc, indspec)
            code += getdecl(childfunc["name"], args, "")
            lenarg = None
            for arg in args.keys():
                if "len" in arg:
                    assert lenarg is None
                    lenarg = arg
            assert lenarg is not None
            code += """  dim3 blocks_per_grid;
dim3 threads_per_block;

if ({0} > 1024) {{
blocks_per_grid = dim3(ceil(({0}) / 1024.0), 1, 1);
threads_per_block = dim3(1024, 1, 1);
}} else {{
blocks_per_grid = dim3(1, 1, 1);
threads_per_block = dim3({0}, 1, 1);
}}
""".format(
                lenarg
            )
            templatetypes = gettemplatetypes(childfunc, templateargs)
            paramnames = getparamnames(args)
            code += (
                " " * 2
                + "cuda"
                + indspec["name"][len("awkward") :]
                + "<"
                + templatetypes
                + "><<<blocks_per_grid, threads_per_block>>>("
                + paramnames
                + ");\n"
            )
            code += " " * 2 + "cudaDeviceSynchronize();\n"
            code += " " * 2 + "return success();\n"
            code += "}\n\n"
    else:
        code += getdecl(indspec["name"], args, "")
        lenarg = None
        for arg in args.keys():
            if "len" in arg:
                assert lenarg is None
                lenarg = arg
        assert lenarg is not None
        code += """  dim3 blocks_per_grid;
dim3 threads_per_block;

if ({0} > 1024) {{
blocks_per_grid = dim3(ceil(({0}) / 1024.0), 1, 1);
threads_per_block = dim3(1024, 1, 1);
}} else {{
blocks_per_grid = dim3(1, 1, 1);
threads_per_block = dim3({0}, 1, 1);
}}
""".format(
            lenarg
        )
        paramnames = getparamnames(args)
        code += (
            " " * 2
            + "cuda"
            + indspec["name"][len("awkward") :]
            + "<<<blocks_per_grid, threads_per_block>>>("
            + paramnames
            + ");\n"
        )
        code += " " * 2 + "cudaDeviceSynchronize();\n"
        code += " " * 2 + "return success();\n"
        code += "}\n\n"
    return code


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("kernelname", nargs="?")
    args = arg_parser.parse_args()
    kernelname = args.kernelname
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "kernelnames.yml")
    ) as infile:
        mainspec = yaml.safe_load(infile)["kernels"]
        code = ""
        for filedir in mainspec.values():
            for relpath in filedir.values():
                with open(
                    os.path.join(CURRENT_DIR, "..", "kernel-specification", relpath)
                ) as specfile:
                    indspec = yaml.safe_load(specfile)[0]
                    if indspec["name"] == kernelname and kernelname in KERNEL_WHITELIST:
                        code = getcode(indspec)
                        break
                    if kernelname is None and indspec["name"] in KERNEL_WHITELIST:
                        code += getcode(indspec)
        print(code)

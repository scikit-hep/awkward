import ast
import os
from collections import OrderedDict

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
KERNEL_WHITELIST = ["awkward_new_Identities"]


def traverse(node):
    if node.__class__.__name__ == "For":
        for subnode in node.body:
            code = traverse(subnode)
    elif node.__class__.__name__ == "Assign":
        assert len(node.targets) == 1
        if node.value.id == "i":
            value = "thread_id"
        else:
            value = node.value.id
        if node.targets[0].slice.value.id == "i":
            index = "thread_id"
        else:
            index = node.targets[0].slice.value.id
        code = node.targets[0].value.id + "[" + index + "] = " + value + ";\n"
    return code


def getbody(pycode):
    code = ""
    tree = ast.parse(pycode).body[0]
    for node in tree.body:
        code += traverse(node)
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
        for childfunc in spec["specializations"]:
            for i in range(len(childfunc["args"])):
                if len(typelist) < i + 1:
                    typelist.append(list(childfunc["args"][i].values())[0])
                else:
                    if typelist[i] != list(childfunc["args"][i].values())[0]:
                        if count == 0:
                            templateargs[list(childfunc["args"][i].keys())[0]] = "T"
                            count += 1
                        else:
                            templateargs[list(childfunc["args"][i].keys())[0]] = "C"
    return templateargs


def getparentargs(templateargs, spec):
    args = OrderedDict()
    if "specializations" in spec.keys():
        for arg in spec["specializations"][0]["args"]:
            argname = list(arg.keys())[0]
            if argname in spec["outparams"]:
                argname = "*" + argname
            if list(arg.keys())[0] in templateargs.keys():
                args[argname] = templateargs[list(arg.keys())[0]]
            elif "len" not in argname and "*" not in argname:
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
    for x in templateargs.values():
        if count == 0:
            templatestring = "typename " + x
            count += 1
        else:
            templatestring = ", typename " + x
    return templatestring


def getdecl(name, parentargs, templatestring, parent=False):
    code = ""
    if templatestring != "":
        code += "<" + templatestring + ">\n"
    if parent:
        code += "__global__\n"
    count = 0
    for key, value in parentargs.items():
        if count == 0:
            params = value + " " + key
            count += 1
        else:
            params += ", " + value + " " + key
    code += "ERROR " + getcudaname(name) + "(" + params + ") {\n"
    return code


def gettemplatetypes(spec, templateargs):
    count = 0
    for arg in childfunc["args"]:
        for argname, typename in arg.items():
            if argname in templateargs.keys():
                if count == 0:
                    code = getctype(typename).replace("*", "")
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


if __name__ == "__main__":
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
                    if indspec["name"] in KERNEL_WHITELIST:
                        templateargs = gettemplateargs(indspec)
                        args = getparentargs(templateargs, indspec)
                        templatestring = gettemplatestring(templateargs)
                        code += getdecl(
                            indspec["name"], args, templatestring, parent=True
                        )
                        code += """  int64_t block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;
"""
                        code += getbody(indspec["definition"])
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
                                templatetypes = gettemplatetypes(
                                    childfunc, templateargs
                                )
                                paramnames = getparamnames(args)
                                code += (
                                    " " * 2
                                    + getcudaname(indspec["name"])
                                    + "<"
                                    + templatetypes
                                    + "><<<blocks_per_grid, threads_per_block>>>("
                                    + paramnames
                                    + ");\n"
                                )
                                code += " " * 2 + "cudaDeviceSynchronize();\n"
                                code += " " * 2 + "return success();\n"
                                code += "}\n\n"
        print(code)

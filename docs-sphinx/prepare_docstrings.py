import re
import os
import ast
import glob

import sphinx.ext.napoleon

config = sphinx.ext.napoleon.Config(napoleon_use_param=True,
                                    napoleon_use_rtype=True)

if not os.path.exists("python"):
    os.mkdir("python")

toctree = []

def tostr(node):
    if isinstance(node, ast.NameConstant):
        return repr(node.value)

    elif isinstance(node, ast.Num):
        return repr(node.n)

    elif isinstance(node, ast.Str):
        return repr(node.s)

    elif isinstance(node, ast.Name):
        return node.id

    elif isinstance(node, ast.Index):
        return tostr(node.value)

    elif isinstance(node, ast.Attribute):
        return tostr(node.value) + "." + node.attr

    elif isinstance(node, ast.Subscript):
        return "{0}[{1}]".format(tostr(node.value), tostr(node.slice))
    
    elif isinstance(node, ast.Slice):
        start = "" if node.lower is None else tostr(node.lower)
        stop  = "" if node.upper is None else tostr(node.upper)
        step  = "" if node.step  is None else tostr(node.step)
        if step == "":
            return "{0}:{1}".format(start, stop)
        else:
            return "{0}:{1}:{2}".format(start, stop, step)

    elif isinstance(node, ast.Call):
        return "{0}({1})".format(tostr(node.func),
                                 ", ".join(tostr(x) for x in node.args))

    elif isinstance(node, ast.List):
        return "[{0}]".format(", ".join(tostr(x) for x in node.elts))

    elif isinstance(node, ast.Set):
        return "{{{0}}}".format(", ".join(tostr(x) for x in node.elts))

    elif isinstance(node, ast.Tuple):
        return "({0})".format(", ".join(tostr(x) for x in node.elts)
                              + ("," if len(node.elts) == 1 else ""))

    elif isinstance(node, ast.Dict):
        return "{{{0}}}".format(", ".join("{0}: {1}".format(tostr(x), tostr(y))
                                      for x, y in zip(node.keys, node.values)))

    elif isinstance(node, ast.Lambda):
        return "lambda {0}: {1}".format(
            ", ".join(x.arg for x in node.args.args),
            tostr(node.body))

    elif isinstance(node, ast.UnaryOp):
        return tostr.op[type(node.op)] + tostr(node.operand)

    elif isinstance(node, ast.BinOp):
        return tostr(node.left) + tostr.op[type(node.op)] + tostr(node.right)

    elif isinstance(node, ast.Compare):
        return tostr(node.left) + "".join(tostr.op[type(x)] + tostr(y)
                                  for x, y in zip(node.ops, node.comparators))

    elif isinstance(node, ast.BoolOp):
        return tostr.op[type(node.op)].join(tostr(x) for x in node.values)

    else:
        raise Exception(ast.dump(node))

tostr.op = {
    ast.And: " and ",
    ast.Or: " or ",
    ast.Add: " + ",
    ast.Sub: " - ",
    ast.Mult: " * ",
    ast.MatMult: " @ ",
    ast.Div: " / ",
    ast.Mod: " % ",
    ast.Pow: "**",
    ast.LShift: " << ",
    ast.RShift: " >> ",
    ast.BitOr: " | ",
    ast.BitXor: " ^ ",
    ast.BitAnd: " & ",
    ast.FloorDiv: " // ",
    ast.Invert: "~",
    ast.Not: "not ",
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Eq: " == ",
    ast.NotEq: " != ",
    ast.Lt: " < ",
    ast.LtE: " <= ",
    ast.Gt: " > ",
    ast.GtE: " >= ",
    ast.Is: " is ",
    ast.IsNot: " is not ",
    ast.In: " in ",
    ast.NotIn: " not in "}

def dosig(node):
    if node is None:
        return "(self)"
    else:
        argnames = [x.arg for x in node.args.args]
        defaults = ["=" + tostr(x) for x in node.args.defaults]
        defaults = [""]*(len(argnames) - len(defaults)) + defaults
        return ", ".join(x + y for x, y in zip(argnames, defaults))

def dodoc(docstring, qualname):
    step1 = docstring.replace("`", "``")
    step2 = re.sub(r"#(ak\.[A-Za-z0-9_\.]+)",
                   r":py:obj:`\1`",
                   step1)
    step3 = re.sub(r"#([A-Za-z0-9_]+)",
                   r":py:meth:`\1 <" + qualname + r".\1>`",
                   step2)
    return str(sphinx.ext.napoleon.GoogleDocstring(step3, config))

def doclass(link, shortname, name, astcls):
    qualname = shortname + "." + name

    init, rest = None, []
    for node in astcls.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init = node
            else:
                rest.append(node)
        elif (isinstance(node, ast.Assign) and
              len(node.targets) == 1 and
              isinstance(node.targets[0], ast.Name)):
            rest.append(node)

    toctree.append(os.path.join("python", qualname + ".rst"))
    with open(toctree[-1], "w") as outfile:
        outfile.write(qualname + "\n" + "-"*len(qualname) + "\n\n")
        outfile.write("Defined in {0}.\n\n".format(link))
        outfile.write(".. py:class:: {0}({1})\n\n".format(qualname,
                                                          dosig(init)))

        docstring = ast.get_docstring(astcls)
        if docstring is not None:
            outfile.write(dodoc(docstring, qualname) + "\n\n")

        for node in rest:
            if isinstance(node, ast.Assign):
                attrtext = "{0}.{1}".format(qualname,
                                            node.targets[0].id)
                outfile.write(".. py:attribute:: " + attrtext + "\n")
                outfile.write("    :value: {0}\n\n".format(tostr(node.value)))

            elif any(isinstance(x, ast.Name) and x.id == "property"
                     for x in node.decorator_list):
                attrtext = "{0}.{1}".format(qualname, node.name)
                outfile.write(".. py:attribute:: " + attrtext + "\n\n")

            elif any(isinstance(x, ast.Attribute) and x.attr == "setter"
                     for x in node.decorator_list):
                pass

            else:
                methodtext = "{0}.{1}({2})".format(qualname,
                                                   node.name,
                                                   dosig(node))
                outfile.write(".. py:method:: " + methodtext + "\n\n")

def dofunction(link, shortname, name, astfcn):
    qualname = shortname + "." + name

    toctree.append(os.path.join("python", qualname + ".rst"))
    with open(toctree[-1], "w") as outfile:
        outfile.write(qualname + "\n" + "-"*len(qualname) + "\n\n")
        outfile.write("Defined in {0}.\n\n".format(link))

        functiontext = "{0}.{1}({2})".format(qualname,
                                             astfcn.name,
                                             dosig(astfcn))
        outfile.write(".. py:function:: " + functiontext + "\n\n")

        docstring = ast.get_docstring(astfcn)
        if docstring is not None:
            outfile.write(dodoc(docstring, qualname) + "\n\n")

for filename in sorted(glob.glob("../src/awkward1/**/*.py", recursive=True),
                       key=lambda x: x.replace("/highlevel", "!")
                                      .replace("/__init__.py", "#")
                                      .replace("/operations", "$")
                                      .replace("/_", "/~")):

    modulename = (filename.replace("../src/", "")
                          .replace("/__init__.py", "")
                          .replace(".py", "")
                          .replace("/", "."))

    shortname = (modulename.replace("awkward1.", "ak.")
                           .replace(".highlevel", "")
                           .replace(".operations.convert", "")
                           .replace(".operations.describe", "")
                           .replace(".operations.structure", "")
                           .replace(".operations.reducers", "")
                           .replace(".behaviors.string", ""))

    link = ("`{0} <https://github.com/scikit-hep/awkward-1.0/blob/"
            "master/{1}>`__".format(modulename, filename.replace("../", "")))

    module = ast.parse(open(filename).read())

    for toplevel in module.body:
        if isinstance(toplevel, ast.ClassDef):
            doclass(link, shortname, toplevel.name, toplevel)
        if isinstance(toplevel, ast.FunctionDef):
            dofunction(link, shortname, toplevel.name, toplevel)

    with open(os.path.join("python", "toctree.txt"), "w") as outfile:
        outfile.write(".. toctree::\n    :hidden:\n\n")
        for x in toctree:
            outfile.write("    " + x + "\n")

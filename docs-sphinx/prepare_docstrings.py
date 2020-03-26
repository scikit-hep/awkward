import re
import os
import ast
import glob

import sphinx.ext.napoleon

config = sphinx.ext.napoleon.Config(napoleon_use_param=True,
                                    napoleon_use_rtype=True)

classes = []
functions = []

def dosig(node):
    if node is None:
        return "(self)"
    else:
        argnames = [x.arg for x in node.args.args]
        defaults = ["=" + repr(eval(compile(ast.Expression(x), "", "eval")))
                      for x in node.args.defaults]
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

def doclass(modulename, shortname, name, astcls):
    qualname = shortname + "." + name

    init, rest = None, []
    for node in astcls.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init = node
            else:
                rest.append(node)

    with open(os.path.join("python", qualname + ".rst"), "w") as outfile:
        outfile.write(name + "\n" + "-"*len(name) + "\n\n")
        outfile.write(".. py:class:: {0}({1})\n\n".format(qualname,
                                                          dosig(init)))
        if modulename != shortname:
            outfile.write(".. py:class:: {0}.{1}\n\n".format(modulename,
                                                             name))

        docstring = ast.get_docstring(astcls)
        if docstring is not None:
            outfile.write(dodoc(docstring, qualname) + "\n\n")

        for node in rest:
            methodtext = "{0}.{1}({2})".format(qualname,
                                               node.name,
                                               dosig(node))
            outfile.write(".. py:method:: " + methodtext + "\n\n")

def dofunction(qual, name, astfcn):
    print(astfcn)
    print(astfcn._fields)

for filename in sorted(glob.glob("../src/awkward1/**/*.py", recursive=True),
                       key=lambda x: x.replace("/highlevel", "!")
                                      .replace("/__init__.py", "#")
                                      .replace("/operations", "$")
                                      .replace("/_", "/~")):

    modulename = (filename.replace("../src/awkward1/", "ak/")
                          .replace("/__init__.py", "")
                          .replace(".py", "")
                          .replace("/", "."))

    shortname = (modulename.replace(".highlevel", "")
                           .replace(".operations.convert", "")
                           .replace(".operations.describe", "")
                           .replace(".operations.structure", "")
                           .replace(".operations.reducers", "")
                           .replace(".behaviors.string", ""))

    module = ast.parse(open(filename).read())

    if "highlevel" not in filename:
        continue

    for toplevel in module.body:
        if isinstance(toplevel, ast.ClassDef):
            doclass(modulename, shortname, toplevel.name, toplevel)
        if isinstance(toplevel, ast.FunctionDef):
            dofunction(modulename, shortname, toplevel.name, toplevel)

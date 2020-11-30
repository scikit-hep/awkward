# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import re
import os
import ast
import glob
import io

import sphinx.ext.napoleon

config = sphinx.ext.napoleon.Config(napoleon_use_param=True,
                                    napoleon_use_rtype=True)

if not os.path.exists("_auto"):
    os.mkdir("_auto")

toctree = ["_auto/changelog.rst"]

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

def dodoc(docstring, qualname, names):
    out = docstring.replace("`", "``")
    out = re.sub(r"<<([^>]*)>>",
                 r"`\1`_",
                 out)
    out = re.sub(r"#(ak\.[A-Za-z0-9_\.]*[A-Za-z0-9_])",
                 r":py:obj:`\1`",
                 out)
    for x in names:
        out = out.replace("#" + x,
                          ":py:meth:`{1} <{0}.{1}>`".format(qualname, x))
    out = re.sub(r"\[([^\]]*)\]\(([^\)]*)\)",
                 r"`\1 <\2>`__",
                 out)
    out = str(sphinx.ext.napoleon.GoogleDocstring(out, config))
    out = re.sub(r"([^\. \t].*\n[ \t]*)((\n    .*[^ \t].*)(\n    .*[^ \t].*|\n[ \t]*)*)",
                 "\\1\n.. code-block:: python\n\n\\2",
                 out)
    out = re.sub(r"(\n:param|^:param)",     "\n    :param",   out)
    out = re.sub(r"(\n:type|^:type)",       "\n    :type",    out)
    out = re.sub(r"(\n:returns|^:returns)", "\n    :returns", out)
    out = re.sub(r"(\n:raises|^:raises)",   "\n    :raises",  out)
    return out

def doclass(link, shortname, name, astcls):
    if name.startswith("_"):
        return

    qualname = shortname + "." + name

    init, rest, names = None, [], []
    for node in astcls.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init = node
            else:
                rest.append(node)
                names.append(node.name)
        elif (isinstance(node, ast.Assign) and
              len(node.targets) == 1 and
              isinstance(node.targets[0], ast.Name)):
            rest.append(node)
            names.append(node.targets[0].id)

    outfile = io.StringIO()
    outfile.write(qualname + "\n" + "-"*len(qualname) + "\n\n")
    outfile.write("Defined in {0}.\n\n".format(link))
    outfile.write(".. py:class:: {0}({1})\n\n".format(qualname, dosig(init)))

    docstring = ast.get_docstring(astcls)
    if docstring is not None:
        outfile.write(dodoc(docstring, qualname, names) + "\n\n")

    for node in rest:
        if isinstance(node, ast.Assign):
            attrtext = "{0}.{1}".format(qualname, node.targets[0].id)
            outfile.write(attrtext + "\n" + "="*len(attrtext) + "\n\n")
            outfile.write(".. py:attribute:: " + attrtext + "\n")
            outfile.write("    :value: {0}\n\n".format(tostr(node.value)))
            docstring = None

        elif any(isinstance(x, ast.Name) and x.id == "property"
                 for x in node.decorator_list):
            attrtext = "{0}.{1}".format(qualname, node.name)
            outfile.write(attrtext + "\n" + "="*len(attrtext) + "\n\n")
            outfile.write(".. py:attribute:: " + attrtext + "\n\n")
            docstring = ast.get_docstring(node)

        elif any(isinstance(x, ast.Attribute) and x.attr == "setter"
                 for x in node.decorator_list):
            docstring = None

        else:
            methodname = "{0}.{1}".format(qualname, node.name)
            methodtext = "{0}({1})".format(methodname, dosig(node))
            outfile.write(methodname + "\n" + "="*len(methodname) + "\n\n")
            outfile.write(".. py:method:: " + methodtext + "\n\n")
            docstring = ast.get_docstring(node)

        if docstring is not None:
            outfile.write(dodoc(docstring, qualname, names) + "\n\n")

    toctree.append(os.path.join("_auto", qualname + ".rst"))
    out = outfile.getvalue()
    if not os.path.exists(toctree[-1]) or open(toctree[-1]).read() != out:
        print("writing", toctree[-1])
        with open(toctree[-1], "w") as outfile:
            outfile.write(out)

def dofunction(link, shortname, name, astfcn):
    if name.startswith("_"):
        return

    qualname = shortname + "." + name

    outfile = io.StringIO()
    outfile.write(qualname + "\n" + "-"*len(qualname) + "\n\n")
    outfile.write("Defined in {0}.\n\n".format(link))

    functiontext = "{0}({1})".format(qualname, dosig(astfcn))
    outfile.write(".. py:function:: " + functiontext + "\n\n")

    docstring = ast.get_docstring(astfcn)
    if docstring is not None:
        outfile.write(dodoc(docstring, qualname, []) + "\n\n")

    out = outfile.getvalue()

    toctree.append(os.path.join("_auto", qualname + ".rst"))
    if not os.path.exists(toctree[-1]) or open(toctree[-1]).read() != out:
        print("writing", toctree[-1])
        with open(toctree[-1], "w") as outfile:
            outfile.write(out)

done_extra = False
for filename in sorted(glob.glob("../src/awkward1/**/*.py", recursive=True),
                       key=lambda x: x.replace("/__init__.py",    "!")
                                      .replace("/highlevel",      "#")
                                      .replace("/operations",     "$")

                                      .replace("/describe.py",    "#")
                                      .replace("/convert.py",     "$")
                                      .replace("/structure.py",   "%")
                                      .replace("/reducers.py",    "&")
                                      .replace("/categorical.py", "'")

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
                           .replace(".behaviors.mixins", "")
                           .replace(".behaviors.categorical", "")
                           .replace(".behaviors.string", ""))

    if modulename == "awkward1.operations.describe":
        toctree.append("ak.behavior.rst")
    elif not done_extra and modulename.startswith("awkward1._"):
        done_extra = True
        toctree.extend(["ak.numba.register.rst",
                        "ak.numexpr.evaluate.rst",
                        "ak.numexpr.re_evaluate.rst",
                        "ak.autograd.elementwise_grad.rst",
                        "ak.layout.Content.rst",
                        "ak.layout.EmptyArray.rst",
                        "ak.layout.NumpyArray.rst",
                        "ak.layout.RegularArray.rst",
                        "ak.layout.ListArray.rst",
                        "ak.layout.ListOffsetArray.rst",
                        "ak.layout.RecordArray.rst",
                        "ak.layout.Record.rst",
                        "ak.layout.IndexedArray.rst",
                        "ak.layout.IndexedOptionArray.rst",
                        "ak.layout.ByteMaskedArray.rst",
                        "ak.layout.BitMaskedArray.rst",
                        "ak.layout.UnmaskedArray.rst",
                        "ak.layout.UnionArray.rst",
                        "ak.layout.VirtualArray.rst",
                        "ak.layout.ArrayGenerator.rst",
                        "ak.layout.SliceGenerator.rst",
                        "ak.layout.ArrayCache.rst",
                        "ak.layout.Iterator.rst",
                        "ak.layout.ArrayBuilder.rst",
                        "ak.layout.Index.rst",
                        "ak.layout.Identities.rst",
                        "ak.types.Type.rst",
                        "ak.types.ArrayType.rst",
                        "ak.types.UnknownType.rst",
                        "ak.types.PrimitiveType.rst",
                        "ak.types.RegularType.rst",
                        "ak.types.ListType.rst",
                        "ak.types.RecordType.rst",
                        "ak.types.OptionType.rst",
                        "ak.types.UnionType.rst",
                        "ak.forms.Form.rst",
                        "ak.forms.EmptyForm.rst",
                        "ak.forms.NumpyForm.rst",
                        "ak.forms.RegularForm.rst",
                        "ak.forms.ListForm.rst",
                        "ak.forms.ListOffsetForm.rst",
                        "ak.forms.RecordForm.rst",
                        "ak.forms.IndexedForm.rst",
                        "ak.forms.IndexedOptionForm.rst",
                        "ak.forms.ByteMaskedForm.rst",
                        "ak.forms.BitMaskedForm.rst",
                        "ak.forms.UnmaskedForm.rst",
                        "ak.forms.UnionForm.rst",
                        "ak.forms.VirtualForm.rst",
                        "ak._io.fromjson.rst",
                        "ak._io.fromroot_nestedvector.rst",
                        ])

    link = ("`{0} <https://github.com/scikit-hep/awkward-1.0/blob/"
            "master/{1}>`__".format(modulename, filename.replace("../", "")))

    module = ast.parse(open(filename).read())

    for toplevel in module.body:
        if isinstance(toplevel, ast.ClassDef):
            doclass(link, shortname, toplevel.name, toplevel)
        if isinstance(toplevel, ast.FunctionDef):
            dofunction(link, shortname, toplevel.name, toplevel)

outfile = io.StringIO()
outfile.write(".. toctree::\n    :hidden:\n\n")
for x in toctree:
    outfile.write("    " + x + "\n")

out = outfile.getvalue()
outfilename = os.path.join("_auto", "toctree.txt")
if not os.path.exists(outfilename) or open(outfilename).read() != out:
    print("writing", outfilename)
    with open(outfilename, "w") as outfile:
        outfile.write(out)

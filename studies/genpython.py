import argparse
import pycparser
import re

def preprocess(filename):
    code = ""
    func = False
    templ = False
    tokens = {}
    templateids = []
    with open(filename, "r") as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1].rstrip() + "\n"
            else:
                line = line.rstrip()
            if line.startswith("#"):
                continue
            if line.startswith("//"):
                continue
            if line.startswith("template") and func is False:
                templ = True
            if "typename" in line:
                iterate = True
                tempids = []
                while iterate:
                    if re.search("typename [^,]*,", line) is not None:
                        tempids.append(line[re.search("typename [^,]*,", line).span()[0]+9:re.search("typename [^,]*,", line).span()[1]-1])
                        line = line[re.search("typename [^,]*,", line).span()[1]:]
                    if re.search("typename [^,]*,", line) is None:
                        iterate = False
                if re.search("typename [^,]*>", line) is not None:
                    tempids.append(line[re.search("typename [^,]*>", line).span()[0]+9:re.search("typename [^,]*>", line).span()[1]-1])
                    line = line[re.search("typename [^,]*>", line).span()[1]:]
                for x in tempids:
                    templateids.append(x + "*")
                for x in tempids:
                    templateids.append(x)
                continue
            if func is True and line.count("{") > 0:
                for _ in range(line.count("{")):
                    parans.append("{")
            if func is False and re.search("\s.*\(", line):
                funcname = re.search("\s.*\(", line).group()[1:-1]
                tokens[funcname] = {"type": line.lstrip().split(" ")[0]}
                line = line.replace(line.split(" ")[0], "int")
                func = True
                parans = []
                code += line
                if line.count("{") > 0:
                    for _ in range(line.count("{")):
                        parans.append("{")
                continue
            if func is True and re.search("<.*>", line) is not None:
                line = line.replace(re.search("<.*>", line).group(), "")
            if func is True and re.search("u?int\d{1,2}_t\*?", line) is not None:
                if "=" not in line and "(" not in line:
                    varname = line[re.search("u?int\d{1,2}_t\*?", line).span()[1] + 1:]
                    varname = re.sub("[\W_]+", "", varname)
                    tokens[funcname][varname] = re.search("u?int\d{1,2}_t\*?", line).group()
                line = line.replace(re.search("u?int\d{1,2}_t\*?", line).group(), "int", 1)
            if func is True and templ is True:
                for x in templateids:
                    if x in line:
                        if line[line.find(x)-1] == " " or line[line.find(x)-1] == "*" or line[line.find(x)-1]:
                            if "=" not in line:
                                varnamestart = line.find(x) + len(x) + 1
                                varnameend = line[varnamestart:].find(",") + varnamestart
                                varname = line[varnamestart:varnameend]
                                tokens[funcname][varname] = x
                            if x.endswith("*"):
                                x = x[:-1]
                            line = line.replace(x, "int")
            code += line
            if func is True and line.count("}") > 0:
                for _ in range(line.count("}")):
                    parans.pop()
                if len(parans) == 0:
                    func = False
                    templ = False
                    templateids = []

    return code,tokens

class FuncBody(object):

    def __init__(self, ast):
        self.ast = ast
        self.code = ""
        self.traverse(self.ast.block_items, 4)

    def traverse(self, item, indent, called=False):
        if item.__class__.__name__ == "list":
            for node in item:
                self.traverse(node, indent)
        elif item.__class__.__name__ == "Return":
            if called:
                return " "*indent + "return {0}".format(self.traverse(item.expr, 0, called=True))
            else:
                self.code += " "*indent + "return {0}\n".format(self.traverse(item.expr, 0, called=True))
        elif item.__class__.__name__ == "Constant":
            constant = " "*indent + "{0}".format(item.value)
            if called:
                return constant
            else:
                self.code += constant
        elif item.__class__.__name__ == "Decl":
            if called:
                return " "*indent + "{0} = {1}".format(item.name, self.traverse(item.init, 0, called=True))
            else:
                self.code += " "*indent + "{0} = {1}\n".format(item.name, self.traverse(item.init, 0, called=True))
        elif item.__class__.__name__ == "Assignment":
            assignstmt = " "*indent + "{0} = {1}\n".format(self.traverse(item.lvalue, 0, called=True), self.traverse(item.rvalue, 0, called=True))
            if called:
                return assignstmt
            else:
                self.code += assignstmt
        elif item.__class__.__name__ == "FuncCall":
            if item.args is not None:
                return " "*indent + "{0}({1})".format(item.name.name, self.traverse(item.args, 0, called=True))
            else:
                return " " * indent + "{0}()".format(item.name.name)
        elif item.__class__.__name__ == "ExprList":
            exprlist = " "*indent
            for i in range(len(item.exprs)):
                if i == 0:
                    exprlist += "{0}".format(self.traverse(item.exprs[i], 0, called=True))
                else:
                    exprlist += ", {0}".format(self.traverse(item.exprs[i], 0, called=True))
            if called:
                return exprlist
            else:
                self.code += exprlist
        elif item.__class__.__name__ == "BinaryOp":
            binaryop = " "*indent + "{0} {1} {2}".format(self.traverse(item.left, 0, called=True), item.op, self.traverse(item.right, 0, called=True))
            if called:
                return binaryop
            else:
                self.code += binaryop
        elif item.__class__.__name__ == "If":
            ifstmt = " "*indent + "if {0}:\n".format(self.traverse(item.cond, 0, called=True))
            ifstmt += " "*(indent + 4) + "{0}\n".format(self.traverse(item.iftrue, 0, called=True))
            if item.iffalse is not None:
                ifstmt += " "*indent + "else:\n"
                ifstmt += "{0}\n".format(self.traverse(item.iffalse, indent + 4, called=True))
            if called:
                return ifstmt
            else:
                self.code += ifstmt
        elif item.__class__.__name__ == "For":
            forstmt = "{0}".format(self.traverse(item.init, indent, called=True))
            forstmt += " "*indent + "while {0}:\n".format(self.traverse(item.cond, 0, called=True))
            for i in range(len(item.stmt.block_items)):
                forstmt += self.traverse(item.stmt.block_items[i], indent+4, called=True) + "\n"
            forstmt += " "*(indent+4) + "{0}\n".format(self.traverse(item.next, 0, called=True))
            if called:
                return forstmt
            else:
                self.code += forstmt
        elif item.__class__.__name__ == "UnaryOp":
            if item.op[1:] == "++":
                unaryop = " "*indent + "{0} = {0} + 1\n".format(item.expr.name)
            elif item.op[1:] == "--":
                unaryop = " " * indent + "{0} = {0} - 1\n".format(item.expr.name)
            elif item.op == "*":
                unaryop = " "*indent + "{0}".format(self.traverse(item.expr, 0, called=True))
            elif item.op == "-":
                unaryop = " "*indent + "-{0}".format(self.traverse(item.expr, 0, called=True))
            else:
                raise NotImplementedError("Please inform the developers about the error")
            if called:
                return unaryop
            else:
                self.code += unaryop
        elif item.__class__.__name__ == "DeclList":
            decllist = " "*indent
            for i in range(len(item.decls)):
                if i == 0:
                    decllist += "{0}".format(self.traverse(item.decls[i], 0, called=True))
                else:
                    decllist += ", {0}".format(self.traverse(item.decls[i], 0, called=True))
            decllist += "\n"
            if called:
                return decllist
            else:
                self.code += decllist
        elif item.__class__.__name__ == "ArrayRef":
            arrayref = " "*indent + "{0}[{1}]".format(self.traverse(item.name, 0, called=True), self.traverse(item.subscript, 0, called=True))
            if called:
                return arrayref
            else:
                self.code += arrayref
        elif item.__class__.__name__ == "Cast":
            cast = " "*indent + "{0}({1})".format(self.traverse(item.to_type, 0, called=True), self.traverse(item.expr, 0, called=True))
            if called:
                return cast
            else:
                self.code += cast
        elif item.__class__.__name__ == "Typename":
            typename = " "*indent + "{0}".format(item.type.type.names[0])
            if called:
                return typename
            else:
                self.code += typename
        elif item.__class__.__name__ == "ID":
            ID = " "*indent + "{0}".format(item.name)
            if called:
                return ID
            else:
                self.code += ID
        elif item.__class__.__name__ == "Compound":
            compound = ""
            for i in range(len(item.block_items)):
                compound += self.traverse(item.block_items[i], indent + 4, called=True) + "\n"
            return compound
        elif item.__class__.__name__ == "EmptyStatement":
            pass
        else:
            raise Exception("Unable to parse {0}".format(item.__class__.__name__))

class FuncDecl(object):

    def __init__(self, ast):
        self.ast = ast
        self.name = ast.name
        self.args = []
        self.returntype = self.ast.type.type.type.names[0]
        self.traverse()

    def traverse(self):
        if self.ast.type.args is not None:
            params = self.ast.type.args.params
            for param in params:
                typename, listcount = self.iterclass(param.type, 0)
                self.args.append({"name": param.name,
                                  "type": typename,
                                  "list": listcount})

    def iterclass(self, obj, count):
        if obj.__class__.__name__ == "IdentifierType":
            return obj.names[0], count
        elif obj.__class__.__name__ == "TypeDecl":
            return self.iterclass(obj.type, count)
        elif obj.__class__.__name__ == "PtrDecl":
            return self.iterclass(obj.type, count+1)

    def arrange_args(self):
        arranged = ""
        for i in range(len(self.args)):
            if i == 0:
                arranged += "{0}".format(self.args[i]["name"])
            else:
                arranged += ", {0}".format(self.args[i]["name"])
        return arranged

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("filename")
args = arg_parser.parse_args()
filename = args.filename

if __name__ == "__main__":
    pfile, tokens = preprocess(filename)
    ast = pycparser.c_parser.CParser().parse(pfile)
    for i in range(len(ast.ext)):
        decl = FuncDecl(ast.ext[i].decl)
        body = FuncBody(ast.ext[i].body)
        print("{0} : {1}".format(decl.name, tokens[decl.name]["type"]))
        print("----------------------------------------------------")
        for x in decl.args:
            brackets = "[]"*x["list"]
            if x["name"] in tokens[decl.name]:
                typename = tokens[decl.name][x["name"]]
            else:
                typename = x["type"]
            if typename.endswith("*"):
                typename = typename[:-1]
            # FIXME
            if typename == "C" or typename == "T":
                typename = "Any"
            print("{0} : {1}{2}".format(x["name"], typename, brackets))
        print()
        print("def {0}({1}):".format(decl.name, decl.arrange_args()))
        print(body.code)
        print()

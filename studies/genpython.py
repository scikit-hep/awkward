import argparse
import pycparser

class FuncBody(object):

    def __init__(self, ast):
        self.ast = ast
        self.code = ""
        self.traverse(self.ast.block_items, 4)
        print(self.code)

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
            self.code += " "*indent + "{0} = {1}\n".format(self.traverse(item.lvalue, 0, called=True), self.traverse(item.rvalue, 0, called=True))
        elif item.__class__.__name__ == "FuncCall":
            return " "*indent + "{0}({1})".format(item.name.name, self.traverse(item.args, 0, called=True))
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
            self.code += "{0}".format(self.traverse(item.init, indent, called=True))
            self.code += " "*indent + "while {0}:\n".format(self.traverse(item.cond, 0, called=True))
            for i in range(len(item.stmt.block_items)):
                self.traverse(item.stmt.block_items[i], indent+4)
            self.code += " "*(indent+4) + "{0}\n".format(self.traverse(item.next, 0, called=True))
        elif item.__class__.__name__ == "UnaryOp":
            if item.op[1:] == "++":
                unaryop = " "*indent + "{0} = {0} + 1".format(item.expr.name)
            elif item.op[1:] == "--":
                unaryop = " " * indent + "{0} = {0} - 1".format(item.expr.name)
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
            arrayref = " "*indent + "{0}[{1}]".format(item.name.name, self.traverse(item.subscript, 0, called=True))
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
        print("def {0}({1}):".format(self.name, self.arrange_args()))

    def traverse(self):
        if self.ast.type.args is not None:
            params = self.ast.type.args.params
            for param in params:
                self.args.append({"name": param.name,
                                  "type": param.type.type.names[0] if param.type.__class__.__name__ == "TypeDecl" else param.type.type.type.names[0],
                                  "list": param.type.__class__.__name__ == "PtrDecl"})

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

ast = pycparser.parse_file(filename)
for i in range(len(ast.ext)):
    FuncDecl(ast.ext[i].decl)
    FuncBody(ast.ext[i].body)

import pycparser

class FuncBody(object):

    def __init__(self, ast):
        self.ast = ast
        self.traverse(self.ast.block_items, 4)
        self.code = ""

    def traverse(self, tree, indent):
        for item in tree:
            if item.__class__.__name__ == "Return":
                self.code += " "*indent + "return {0}\n".format(self.traverse(item.expr, 0))
            elif item.__class__.__name__ == "Constant":
                self.code += " "*indent + "{0}".format(item.value)
            elif item.__class__.__name__ == "Decl":
                self.code += " "*indent + "{0} = {1}\n".format(item.name, self.traverse(item.init, 0))
            elif item.__class__.__name__ == "Assignment":
                self.code += " "*indent + "{0} = {1}\n".format(item.lvalue.name, self.traverse(item.rvalue, 0))
            elif item.__class__.__name__ == "FuncCall":
                self.code += " "*indent + "{0}({1})\n".format(item.name.name, self.traverse(item.args, 0))
            elif item.__class__.__name__ == "ExprList":
                self.code += " "*indent
                for i in range(len(item.exprs)):
                    if i == 0:
                        self.code += "{0}".format(self.traverse(item.exprs[i], 0))
                    else:
                        self.code += ", {0}".format(self.traverse(item.exprs[i], 0))
            elif item.__class__.__name__ == "BinaryOp":
                self.code += " "*indent + "{0} {1} {2}".format(self.traverse(item.left, 0), item.op, self.traverse(item.right, 0))
            elif item.__class__.__name__ == "If":
                self.code += " "*indent + "if {0}:\n".format(self.traverse(item.cond, 0))
                self.code += " "*(indent + 4) + "{0}\n".format(self.traverse(item.iftrue, 0))
                if item.iffalse is not None:
                    self.code += " "*indent + "else:\n"
                    self.code += " "*(indent + 4) + "{0}".format(self.traverse(item.iffalse, 0))
            elif item.__class__.__name__ == "For":
                self.code += self.traverse(item.init, indent)
                self.code += " "*indent + "while {0}:\n".format(self.traverse(item.cond, 0))
                for i in range(len(item.stmt.block_items)):
                    self.code += self.traverse(item.stmt.block_items[i], indent+4)+"\n"
                self.code += " "*(indent+4) + "{0}\n".format(self.traverse(item.next, 0))
            elif item.__class__.__name__ == "UnaryOp":
                raise NotImplementedError
            elif item.__class__.__name__ == "DeclList":
                self.code += " "*indent
                for i in range(len(item.decls)):
                    if i == 0:
                        self.code += "{0}".format(self.traverse(item.decls[i], 0))
                    else:
                        self.code += ", {0}".format(self.traverse(item.decls[i], 0))
            elif item.__class__.__name__ == "EmptyStatement":
                pass
            else:
                raise Exception("Unable to parse")

class FuncDecl(object):

    def __init__(self, ast):
        self.ast = ast
        self.name = ast.name
        self.args = []
        self.returntype = self.ast.type.type.type.names[0]
        self.traverse()

    def traverse(self):
        params = self.ast.type.args.params
        for param in params:
            self.args.append({"name": param.name,
                              "type": param.type.type.names[0] if param.type.__class__.__name__ == "TypeDecl" else param.type.type.type.names[0],
                              "list": param.type.__class__.__name__ == "PtrDecl"})

ast = pycparser.parse_file("awkwardsample.c")
funcdecl = FuncDecl(ast.ext[0].decl)
funcbody = FuncBody(ast.ext[0].body)


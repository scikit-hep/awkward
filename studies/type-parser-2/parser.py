import json

import lark

grammar = """
start: json



json: json_object
    | json_array
    | ESCAPED_STRING -> string
    | SIGNED_NUMBER  -> number
    | "true"         -> true
    | "false"        -> false
    | "null"         -> null

json_array:  "[" [json ("," json)*] "]"
json_object: "{" [json_pair ("," json_pair)*] "}"
json_pair:   ESCAPED_STRING ":" json

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS
"""

class Transformer:
    def start(self, args):
        return args[0]

    def json(self, args):
        return args[0]

    def json_object(self, args):
        return dict(args)

    def json_pair(self, args):
        return (json.loads(args[0]), args[1])

    def json_array(self, args):
        return list(args)

    def string(self, args):
        return json.loads(args[0])

    def number(self, args):
        try:
            return int(args[0])
        except ValueError:
            return float(args[0])

    def true(self, args):
        return True

    def false(self, args):
        return False

    def null(self, args):
        return None

parser = lark.Lark(grammar, parser="lalr", transformer=Transformer())

datashape_string = """
{"this": ["is", "json", true, false, null, 3.14, 123]}
"""

print(parser.parse(datashape_string))

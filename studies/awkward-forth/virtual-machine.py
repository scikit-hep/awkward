import math
import re
import subprocess
import sys

import numpy as np


class GrowableBuffer:
    def __init__(self, dtype, initial=1024, resize=1.5):
        self.buffer = np.full(initial, 999, dtype=dtype)
        self.length = 0
        self.resize = resize

    def __str__(self):
        return str(self.buffer[:self.length])

    def __repr__(self):
        return "<GrowableBuffer {0}>".format(
            str(self).replace("\n", "\n                ")
        )

    def extend(self, values):
        length = len(self.buffer)
        while self.length + len(values) > length:
            length = int(math.ceil(length * self.resize))

        if length > len(self.buffer):
            replacement = np.full(length, 999, dtype=self.buffer.dtype)
            replacement[:len(self.buffer)] = self.buffer
            self.buffer = replacement

        self.buffer[self.length : self.length + len(values)] = values
        self.length += len(values)

    def append(self, value):
        self.extend([value])


class Stack:
    def __init__(self, dtype=np.dtype(np.int64), length=1024):
        self.buffer = np.full(length, 999, dtype=dtype)
        self.pointer = 0

    def __str__(self):
        return " ".join([str(x) for x in self.buffer[:self.pointer]] + ["<- top"])

    def __repr__(self):
        return "<Stack {0}>".format(str(self))

    def push(self, num):
        if self.pointer >= len(self.buffer):
            raise ValueError("stack overflow")
        self.buffer[self.pointer] = num
        self.pointer += 1

    def pop(self):
        if self.pointer <= 0:
            raise ValueError("stack underflow")
        self.pointer -= 1
        return self.buffer[self.pointer]

    def tolist(self):
        return self.buffer[:self.pointer].tolist()


class VirtualMachine:
    def __init__(self):
        # The dictionary contains user-defined functions as lists of instructions.
        self.dictionary_names = {}
        self.dictionary = []

    def compile(self, source):
        tokenized = re.split("[ \r\t\v\f]", source.replace("\n", " \n "))

        variable_names = []
        input_names = []
        output_names = []
        output_dtypes = []

        def as_integer(word):
            if word.lstrip().startswith("0x"):
                try:
                    return int(word, 16)
                except ValueError:
                    return None
            else:
                try:
                    return int(word)
                except ValueError:
                    return None

        def parse(defn, start, stop, instructions, exitdepth, dodepth):
            pointer = start
            while pointer < stop:
                word = tokenized[pointer]

                if word == "(":
                    # Simply skip the parenthesized text: it's a comment.
                    substop = pointer
                    nesting = 1
                    while nesting > 0:
                        substop += 1
                        if substop >= stop:
                            raise ValueError("'(' is missing its closing ')'")
                        if tokenized[substop] == "(":
                            nesting += 1
                        elif tokenized[substop] == ")":
                            nesting -= 1
                    pointer = substop + 1

                elif word == "\\":
                    # Modern, backslash-to-end-of-line comments.
                    substop = pointer
                    while substop < stop and tokenized[substop] != "\n":
                        substop += 1
                    pointer = substop + 1

                elif word == "\n":
                    # End-of-line needs to be a token to parse the above.
                    pointer += 1

                elif word == "":
                    # Remove leading or trailing blanks.
                    pointer += 1

                elif word == ":":
                    if pointer + 1 >= stop or tokenized[pointer + 1] == ";":
                        raise ValueError("missing name in word definition")
                    name = tokenized[pointer + 1]

                    if as_integer(name) is not None or Builtin.reserved(name):
                        raise ValueError(
                            "user-defined words should not overshadow a builtin word: "
                            + repr(name)
                        )

                    substart = pointer + 2
                    substop = pointer + 1
                    nesting = 1
                    while nesting > 0:
                        substop += 1
                        if substop >= stop:
                            raise ValueError("definition is missing its closing ';'")
                        if tokenized[substop] == ":":
                            nesting += 1
                        elif tokenized[substop] == ";":
                            nesting -= 1

                    # Add the new word to the dictionary before parsing it
                    # so that recursive functions can be defined.
                    instruction = len(Builtin.lookup) + len(self.dictionary)
                    self.dictionary_names[name] = instruction

                    # Now parse the subroutine and add it to the dictionary.
                    self.dictionary.append([])
                    parse(name, substart, substop, self.dictionary[-1], 0, 0)

                    pointer = substop + 1

                elif word == "recurse":
                    if defn is None:
                        raise ValueError("'recurse' can only be used in a ':' .. ';' definition")
                    instructions.append(self.dictionary_names[defn])
                    pointer += 1

                elif word == "variable":
                    if pointer + 1 >= stop:
                        raise ValueError("missing name in variable definition")
                    name = tokenized[pointer + 1]
                    pointer += 2

                    if as_integer(name) is not None or Builtin.reserved(name):
                        raise ValueError(
                            "user-defined variables should not overshadow a builtin word: "
                            + repr(name)
                        )

                    if name in variable_names:
                        raise ValueError("variable name {0} redefined".format(repr(name)))

                    variable_names.append(name)

                elif word == "input":
                    if pointer + 1 >= stop:
                        raise ValueError("missing name in input definition")
                    name = tokenized[pointer + 1]
                    pointer += 2

                    if as_integer(name) is not None or Builtin.reserved(name):
                        raise ValueError(
                            "user-defined inputs should not overshadow a builtin word: "
                            + repr(name)
                        )

                    if name in input_names:
                        raise ValueError("input name {0} redefined".format(repr(name)))

                    input_names.append(name)

                elif word == "output":
                    if pointer + 2 >= stop:
                        raise ValueError("missing name or dtype in output definition")
                    name = tokenized[pointer + 1]
                    dtype = tokenized[pointer + 2]
                    pointer += 3

                    if as_integer(name) is not None or Builtin.reserved(name):
                        raise ValueError(
                            "user-defined outputs should not overshadow a builtin word: "
                            + repr(name)
                        )

                    if name in output_names:
                        raise ValueError("output name {0} redefined".format(repr(name)))

                    if dtype not in Builtin.dtypes:
                        raise ValueError(
                            "output dtype must be one of: ".format(", ".join(Builtin.dtypes))
                        )

                    output_names.append(name)
                    output_dtypes.append(dtype)

                elif word in variable_names:
                    if pointer + 1 >= stop or tokenized[pointer + 1] not in ("!", "+!", "@"):
                        raise ValueError("missing '!', '+!', or '@' after variable name")
                    if tokenized[pointer + 1] == "!":
                        instructions.append(Builtin.PUT.as_integer)
                    elif tokenized[pointer + 1] == "+!":
                        instructions.append(Builtin.INC.as_integer)
                    else:
                        instructions.append(Builtin.GET.as_integer)
                    instructions.append(variable_names.index(word))
                    pointer += 2

                elif word in input_names:
                    if pointer + 1 >= stop or tokenized[pointer + 1] not in Builtin.ptypes:
                        raise ValueError(
                            "missing parser word after input name; must be one of: "
                            + ", ".join(Builtin.ptypes)
                        )
                    ptype = tokenized[pointer + 1]
                    pointer += 2

                    direct = False
                    output = None
                    if pointer + 1 < stop and tokenized[pointer + 1] in output_names:
                        direct = True
                        output = tokenized[pointer + 1]
                        pointer += 1

                    instructions.append(Builtin.as_parser(ptype, direct))
                    instructions.append(input_names.index(word))
                    if direct:
                        instructions.append(output_names.index(output))

                elif word == "if":
                    substart = pointer + 1
                    subelse = -1
                    substop = pointer
                    nesting = 1
                    while nesting > 0:
                        substop += 1
                        if substop >= stop:
                            raise ValueError("'if' is missing its closing 'then'")
                        if tokenized[substop] == "if":
                            nesting += 1
                        elif tokenized[substop] == "then":
                            nesting -= 1
                        elif tokenized[substop] == "else" and nesting == 1:
                            subelse = substop

                    if subelse == -1:
                        # Add the consequent to the dictionary so that it can be used
                        # without special instruction pointer manipulation at runtime.
                        consequent = len(Builtin.lookup) + len(self.dictionary)

                        # Now add the consequent to the dictionary.
                        self.dictionary.append([])
                        parse(defn, substart, substop, self.dictionary[-1], exitdepth + 1, dodepth)

                        instructions.append(Builtin.IF.as_integer)
                        instructions.append(consequent)
                        pointer = substop + 1

                    else:
                        # Same as above, except that this is an 'if .. else .. then'.
                        consequent = len(Builtin.lookup) + len(self.dictionary)
                        self.dictionary.append([])
                        parse(defn, substart, subelse, self.dictionary[-1], exitdepth + 1, dodepth)

                        alternate = len(Builtin.lookup) + len(self.dictionary)
                        self.dictionary.append([])
                        parse(defn, subelse + 1, substop, self.dictionary[-1], exitdepth + 1, dodepth)

                        instructions.append(Builtin.IF_ELSE.as_integer)
                        instructions.append(consequent)
                        instructions.append(alternate)
                        pointer = substop + 1

                elif word == "do":
                    substart = pointer + 1
                    substop = pointer
                    is_step = False
                    nesting = 1
                    while nesting > 0:
                        substop += 1
                        if substop >= stop:
                            raise ValueError("'do' is missing its closing 'loop'")
                        if tokenized[substop] == "do":
                            nesting += 1
                        elif tokenized[substop] == "loop":
                            nesting -= 1
                        elif tokenized[substop] == "+loop":
                            if nesting == 1:
                                is_step = True
                            nesting -= 1

                    # Add the loop body to the dictionary so that it can be used
                    # without special instruction pointer manipulation at runtime.
                    body = len(Builtin.lookup) + len(self.dictionary)

                    # Now add the loop body to the dictionary.
                    self.dictionary.append([])
                    parse(defn, substart, substop, self.dictionary[-1], exitdepth + 1, dodepth + 1)

                    if is_step:
                        instructions.append(Builtin.DO_STEP.as_integer)
                        instructions.append(body)
                    else:
                        instructions.append(Builtin.DO.as_integer)
                        instructions.append(body)
                    pointer = substop + 1

                elif word == "begin":
                    substart = pointer + 1
                    substop = pointer
                    is_again = False
                    subwhile = -1
                    nesting = 1
                    while nesting > 0:
                        substop += 1
                        if substop >= stop:
                            raise ValueError(
                                "'begin' is missing its closing 'until' or 'while' .. 'repeat'"
                            )
                        if tokenized[substop] == "begin":
                            nesting += 1
                        elif tokenized[substop] == "until":
                            nesting -= 1
                        elif tokenized[substop] == "again":
                            if nesting == 1:
                                is_again = True
                            nesting -= 1
                        elif tokenized[substop] == "while":
                            if nesting == 1:
                                subwhile = substop
                            nesting -= 1
                            subnesting = 1
                            while subnesting > 0:
                                substop += 1
                                if substop >= stop:
                                    raise ValueError("'while' is missing its closing 'repeat'")
                                if tokenized[substop] == "while":
                                    subnesting += 1
                                elif tokenized[substop] == "repeat":
                                    subnesting -= 1

                    if is_again:
                        # Define the 'begin' .. 'until' statements as an instruction.
                        body = len(Builtin.lookup) + len(self.dictionary)

                        # Now add it to the dictionary.
                        self.dictionary.append([])
                        parse(defn, substart, substop, self.dictionary[-1], exitdepth + 1, dodepth)

                        instructions.append(body)
                        instructions.append(Builtin.AGAIN.as_integer)

                    elif subwhile == -1:
                        # Define the 'begin' .. 'until' statements as an instruction.
                        body = len(Builtin.lookup) + len(self.dictionary)

                        # Now add it to the dictionary.
                        self.dictionary.append([])
                        parse(defn, substart, substop, self.dictionary[-1], exitdepth + 1, dodepth)

                        instructions.append(body)
                        instructions.append(Builtin.UNTIL.as_integer)

                    else:
                        # Define the 'begin' .. 'repeat' statements.
                        unconditional = len(Builtin.lookup) + len(self.dictionary)
                        self.dictionary.append([])
                        parse(defn, substart, subwhile, self.dictionary[-1], exitdepth + 1, dodepth)

                        # Define the 'repeat' .. 'until' statements.
                        conditional = len(Builtin.lookup) + len(self.dictionary)
                        self.dictionary.append([])
                        parse(defn, subwhile + 1, substop, self.dictionary[-1], exitdepth + 1, dodepth)

                        instructions.append(unconditional)
                        instructions.append(Builtin.WHILE.as_integer)
                        instructions.append(conditional)

                    pointer = substop + 1

                elif word == "exit":
                    instructions.append(Builtin.EXIT.as_integer)
                    instructions.append(exitdepth)
                    pointer += 1

                elif word in Builtin.lookup:
                    if word == "i" and dodepth < 1:
                        raise ValueError("'i' can only be used in a 'do' loop")
                    elif word == "j" and dodepth < 2:
                        raise ValueError("'j' can only be used in a nested 'do' loop")
                    elif word == "k" and dodepth < 3:
                        raise ValueError("'k' can only be used in a doubly nested 'do' loop")

                    instructions.append(Builtin.lookup[word].as_integer)
                    pointer += 1

                else:
                    instruction = self.dictionary_names.get(word)
                    num = as_integer(word)

                    if instruction is not None:
                        instructions.append(instruction)
                        pointer += 1

                    elif num is not None:
                        instructions.append(Builtin.LITERAL.as_integer)
                        instructions.append(num)
                        pointer += 1

                    else:
                        raise ValueError(
                            "unrecognized or wrong context for word: " + repr(word)
                        )

            if len(self.dictionary) >= np.iinfo(np.int32).max:
                raise ValueError("source code defines too many instructions")

        instructions = []
        parse(None, 0, len(tokenized), instructions, 0, 0)
        return instructions, variable_names, input_names, output_names, output_dtypes

    def run(self, instructions, variables, inputs, output_dtypes=[], variable_names=[], verbose=False):
        # Create the stack.
        stack = Stack()

        # Ensure that the inputs are raw bytes.
        inputs = [np.frombuffer(x, dtype="u1") for x in inputs]
        input_pointers = [0 for x in inputs]

        # Create the outputs.
        outputs = [GrowableBuffer(x) for x in output_dtypes]

        if verbose:
            print("{0:20s} | {1}".format("instruction", "stack before instruction"))
            print("---------------------+-------------------------")

            def printout(indent, instruction, showstack):
                if showstack:
                    showstack = str(stack)
                else:
                    showstack = ""
                print("{0:20s} | {1}".format("  " * indent + str(instruction), showstack))

        # Create a stack of instruction pointers to avoid native function calls.
        dictionary = [np.array(x, np.int32) for x in self.dictionary + [instructions]]
        which = [len(self.dictionary)]
        where = [0]
        skip = [0]

        # The do .. loop stack is a different stack.
        do_depth = []
        do_start = []
        do_stop = []
        do_step = []
        do_i = []

        # Run through the instructions until the first stack layer reaches its end.
        while len(which) > 0:
            while where[-1] < len(dictionary[which[-1]]):
                instruction = dictionary[which[-1]][where[-1]]

                if len(do_depth) == 0 or do_depth[-1] != len(which):
                    # Normal operation: step forward one instruction.
                    where[-1] += 1

                elif do_i[-1] >= do_stop[-1]:
                    # End a 'do' loop.
                    if verbose:
                        printout(len(which) - 1, "do: {0} (end)".format(do_i[-1]), False)
                    do_depth.pop()
                    do_start.pop()
                    do_stop.pop()
                    do_step.pop()
                    do_i.pop()
                    where[-1] += 1
                    continue

                else:
                    # Step forward in a DO loop.
                    if verbose:
                        printout(len(which) - 1, "do: {0}".format(do_i[-1]), False)

                if skip[-1] != 0:
                    # Skip over the alternate ('else' clause) of an 'if' block.
                    # Or skip backwards in a 'begin' .. loop.
                    where[-1] += skip[-1]
                skip[-1] = 0

                if Builtin.is_parser(instruction):
                    HERE



                elif instruction == Builtin.LITERAL.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, num, True)
                    stack.push(num)

                elif instruction == Builtin.PUT.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, "! " + num, True)
                    variables[num] = stack.pop()

                elif instruction == Builtin.INC.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, "+! " + num, True)
                    variables[num] += stack.pop()

                elif instruction == Builtin.GET.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, "@ " + num, True)
                    stack.push(variables[num])

                elif instruction == Builtin.IF.as_integer:
                    if verbose:
                        printout(len(which) - 1, "if", True)
                    if stack.pop() == 0:
                        # False, so then skip over the next instruction.
                        where[-1] += 1
                    else:
                        # True, so do the next instruction (i.e. normal flow).
                        if verbose:
                            printout(len(which) - 1, "then", False)

                elif instruction == Builtin.IF_ELSE.as_integer:
                    if verbose:
                        printout(len(which) - 1, "if", True)
                    if stack.pop() == 0:
                        if verbose:
                            printout(len(which) - 1, "else", False)
                        # False, so then skip over the next instruction but do the one after that.
                        where[-1] += 1
                    else:
                        if verbose:
                            printout(len(which) - 1, "then", False)
                        # True, so do the next instruction but skip the one after that.
                        skip[-1] = 1

                elif instruction == Builtin.DO.as_integer:
                    do_depth.append(len(which))
                    do_start.append(stack.pop())
                    do_stop.append(stack.pop())
                    do_step.append(False)
                    do_i.append(do_start[-1])

                elif instruction == Builtin.DO_STEP.as_integer:
                    do_depth.append(len(which))
                    do_start.append(stack.pop())
                    do_stop.append(stack.pop())
                    do_step.append(True)
                    do_i.append(do_start[-1])

                elif instruction == Builtin.AGAIN.as_integer:
                    if verbose:
                        printout(len(which) - 1, "again", False)
                    # Go back and do the body again.
                    where[-1] -= 2

                elif instruction == Builtin.UNTIL.as_integer:
                    if verbose:
                        printout(len(which) - 1, "until", True)
                    if stack.pop() == 0:
                        # False, so go back and do the body again.
                        where[-1] -= 2

                elif instruction == Builtin.WHILE.as_integer:
                    if verbose:
                        printout(len(which) - 1, "while", True)
                    if stack.pop() == 0:
                        # False, so skip over the conditional body.
                        where[-1] += 1
                    else:
                        # True, so do the next instruction but skip back after that.
                        skip[-1] = -3

                elif instruction == Builtin.EXIT.as_integer:
                    if verbose:
                        printout(len(which) - 1, "exit", False)
                    exitdepth = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    for i in range(exitdepth):
                        which.pop()
                        where.pop()
                        skip.pop()
                    while len(do_depth) != 0 and do_depth[-1] >= len(which):
                        do_depth.pop()
                        do_start.pop()
                        do_stop.pop()
                        do_step.pop()
                        do_i.pop()
                    break

                elif instruction == Builtin.I.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    if len(do_i) < 1:
                        stack.push(0)
                    else:
                        stack.push(do_i[-1])

                elif instruction == Builtin.J.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    if len(do_i) < 2:
                        stack.push(0)
                    else:
                        stack.push(do_i[-2])

                elif instruction == Builtin.K.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    if len(do_i) < 3:
                        stack.push(0)
                    else:
                        stack.push(do_i[-3])

                elif instruction == Builtin.DUP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    value = stack.pop()
                    stack.push(value)
                    stack.push(value)

                elif instruction == Builtin.DROP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.pop()

                elif instruction == Builtin.SWAP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(a)
                    stack.push(b)

                elif instruction == Builtin.OVER.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(b)
                    stack.push(a)
                    stack.push(b)

                elif instruction == Builtin.ROT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b, c = stack.pop(), stack.pop(), stack.pop()
                    stack.push(b)
                    stack.push(a)
                    stack.push(c)

                elif instruction == Builtin.NIP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(a)

                elif instruction == Builtin.TUCK.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(a)
                    stack.push(b)
                    stack.push(a)

                elif instruction == Builtin.ADD.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() + stack.pop())

                elif instruction == Builtin.SUB.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(b - a)

                elif instruction == Builtin.MUL.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() * stack.pop())

                elif instruction == Builtin.DIV.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(b // a)

                elif instruction == Builtin.MOD.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(b % a)

                elif instruction == Builtin.DIVMOD.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    c, d = divmod(b, a)
                    stack.push(d)
                    stack.push(c)

                elif instruction == Builtin.LSHIFT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop(), stack.pop()
                    stack.push(b << a)

                elif instruction == Builtin.RSHIFT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = stack.pop().view(np.uint64), stack.pop().view(np.uint64)
                    stack.push(b >> a)

                elif instruction == Builtin.ABS.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(abs(stack.pop()))

                elif instruction == Builtin.MIN.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(min(stack.pop(), stack.pop()))

                elif instruction == Builtin.MAX.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(max(stack.pop(), stack.pop()))

                elif instruction == Builtin.NEGATE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-stack.pop())

                elif instruction == Builtin.ADD1.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() + 1)

                elif instruction == Builtin.SUB1.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() - 1)

                elif instruction == Builtin.EQ0.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() == 0 else 0)

                elif instruction == Builtin.EQ.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() == stack.pop() else 0)

                elif instruction == Builtin.NE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() != stack.pop() else 0)

                elif instruction == Builtin.GT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() < stack.pop() else 0)

                elif instruction == Builtin.GE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() <= stack.pop() else 0)

                elif instruction == Builtin.LT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() > stack.pop() else 0)

                elif instruction == Builtin.LE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1 if stack.pop() >= stack.pop() else 0)

                elif instruction == Builtin.AND.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() & stack.pop())

                elif instruction == Builtin.OR.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() | stack.pop())

                elif instruction == Builtin.XOR.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(stack.pop() ^ stack.pop())

                elif instruction == Builtin.INVERT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(~stack.pop())

                elif instruction == Builtin.FALSE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(0)

                elif instruction == Builtin.TRUE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    stack.push(-1)

                elif instruction >= len(Builtin.lookup):
                    if verbose:
                        for name, value in self.dictionary_names.items():
                            if value == instruction:
                                printout(len(which) - 1, name, False)
                                break
                    which.append(instruction - len(Builtin.lookup))
                    where.append(0)
                    skip.append(0)

                else:
                    raise AssertionError(
                        "unrecognized machine code: {0}".format(instruction)
                    )

            which.pop()
            where.pop()
            skip.pop()

            if len(do_depth) > 0 and do_depth[-1] == len(which):
                if do_step[-1]:
                    do_i[-1] += stack.pop()
                else:
                    do_i[-1] += 1

        if verbose:
            print("{0:20s} | {1}".format("", str(stack)))

        self.stack = stack
        return outputs

    def do(self, source, inputs=[], verbose=False):
        if verbose:
            print("do: {0}".format(repr(source)))

        (
            instructions, variable_names, input_names, output_names, output_dtypes
        ) = self.compile(source)

        variables = [0 for x in variable_names]

        self.outputs = self.run(
            instructions,
            variables,
            inputs,
            output_dtypes=output_dtypes,
            variable_names=variable_names,
            verbose=verbose,
        )

        self.variables = dict(zip(variable_names, variables))


class Builtin:
    lookup = {}

    def __init__(self, word=None):
        if word is None:
            word = len(self.lookup)   # just make unique values that aren't strings
        self.word = word
        self.as_integer = len(self.lookup)
        self.lookup[word] = self

    @staticmethod
    def word(integer):
        for word, instruction in Builtin.lookup.items():
            if integer == instruction.as_integer:
                return word
        else:
            raise KeyError("not found: {0}".format(integer))

    dtypes = [
        "bool", "int8", "int16", "int32", "int64"
    ] + [
        "uint8", "uint16", "uint32", "uint64", "float32", "float64"
    ]
    ptypes = [
        n + b + t + "->" for n in ("", "#") for b in ("", "!") for t in "?bhiqnBHIQNfd"
        if not (t == "?" and b == "!")
        and not (t == "b" and b == "!")
        and not (t == "B" and b == "!")
    ]

    @staticmethod
    def reserved(word):
        return (
            word in ["(", ")", "\\", ":", ";", "variable", "!", "+!", "@"]
            or word in ["if", "then", "else"]
            or word in ["do", "loop", "+loop"]
            or word in ["begin", "again", "until", "while", "repeat"]
            or word in ["exit", "recurse"]
            or word in ["input", "output"]
            or word in Builtin.dtypes
            or word in Builtin.ptypes
            or word in Builtin.lookup
        )

    @staticmethod
    def is_parser(word):
        return bool(word & 0x80000000)

    @staticmethod
    def is_direct(word):
        return bool(word & 0x40000000)

    @staticmethod
    def is_repeated(word):
        return bool(word & 0x20000000)

    @staticmethod
    def is_bigendian(word):
        return bool(word & 0x10000000)

    PARSER_BOOL = 0
    PARSER_INT8 = 1
    PARSER_INT16 = 2
    PARSER_INT32 = 3
    PARSER_INT64 = 4
    PARSER_SSIZE = 5
    PARSER_UINT8 = 6
    PARSER_UINT16 = 7
    PARSER_UINT32 = 8
    PARSER_UINT64 = 9
    PARSER_USIZE = 10
    PARSER_FLOAT32 = 11
    PARSER_FLOAT64 = 12

    @staticmethod
    def as_parser(ptype, direct):
        out = np.int32(0x80000000)
        if direct:
            out |= 0x40000000
        if ptype.startswith("#"):
            out |= 0x20000000
            ptype = ptype[1:]
        if ptype.startswith("!"):
            out |= 0x10000000
            ptype = ptype[1:]
        if ptype[0] == "?":
            out |= PARSER_BOOL
        elif ptype[0] == "b":
            out |= PARSER_INT8
        elif ptype[0] == "h":
            out |= PARSER_INT16
        elif ptype[0] == "i":
            out |= PARSER_INT32
        elif ptype[0] == "q":
            out |= PARSER_INT64
        elif ptype[0] == "n":
            out |= PARSER_SSIZE
        elif ptype[0] == "B":
            out |= PARSER_UINT8
        elif ptype[0] == "H":
            out |= PARSER_UINT16
        elif ptype[0] == "I":
            out |= PARSER_UINT32
        elif ptype[0] == "Q":
            out |= PARSER_UINT64
        elif ptype[0] == "N":
            out |= PARSER_USIZE
        elif ptype[0] == "f":
            out |= PARSER_FLOAT32
        elif ptype[0] == "d":
            out |= PARSER_FLOAT64
        else:
            raise AssertionError(ptype)
        return out

Builtin.LITERAL = Builtin()
Builtin.PUT = Builtin()
Builtin.INC = Builtin()
Builtin.GET = Builtin()
Builtin.IF = Builtin()
Builtin.IF_ELSE = Builtin()
Builtin.DO = Builtin()
Builtin.DO_STEP = Builtin()
Builtin.AGAIN = Builtin()
Builtin.UNTIL = Builtin()
Builtin.WHILE = Builtin()
Builtin.EXIT = Builtin()
Builtin.I = Builtin("i")
Builtin.J = Builtin("j")
Builtin.K = Builtin("k")
Builtin.DUP = Builtin("dup")
Builtin.DROP = Builtin("drop")
Builtin.SWAP = Builtin("swap")
Builtin.OVER = Builtin("over")
Builtin.ROT = Builtin("rot")
Builtin.NIP = Builtin("nip")
Builtin.TUCK = Builtin("tuck")
Builtin.ADD = Builtin("+")
Builtin.SUB = Builtin("-")
Builtin.MUL = Builtin("*")
Builtin.DIV = Builtin("/")
Builtin.MOD = Builtin("mod")
Builtin.DIVMOD = Builtin("/mod")
Builtin.LSHIFT = Builtin("lshift")
Builtin.RSHIFT = Builtin("rshift")
Builtin.ABS = Builtin("abs")
Builtin.MIN = Builtin("min")
Builtin.MAX = Builtin("max")
Builtin.NEGATE = Builtin("negate")
Builtin.ADD1 = Builtin("1+")
Builtin.SUB1 = Builtin("1-")
Builtin.EQ0 = Builtin("0=")
Builtin.EQ = Builtin("=")
Builtin.NE = Builtin("<>")
Builtin.GT = Builtin(">")
Builtin.GE = Builtin(">=")
Builtin.LT = Builtin("<")
Builtin.LE = Builtin("<=")
Builtin.AND = Builtin("and")
Builtin.OR = Builtin("or")
Builtin.XOR = Builtin("xor")
Builtin.INVERT = Builtin("invert")
Builtin.FALSE = Builtin("false")
Builtin.TRUE = Builtin("true")


vm = VirtualMachine()
vm.do("3 ( whatever ) 2 + 2 *")
vm.do("3 ( whatever )\n2 + 2 *")
vm.do("3 \\ whatever\n2 + 2 *")
vm.do(": foo 3 2 + ; foo")
vm.do(": foo : bar 1 + ; 3 2 + ; foo bar")
vm.do("1 2 3 dup")
vm.do("1 2 3 drop")
vm.do("1 2 3 4 swap")
vm.do("1 2 3 over")
vm.do("1 2 3 rot")
vm.do(": foo -1 if 3 2 + else 10 20 * then ; foo 999")
vm.do(": foo 0 if 3 2 + else 10 20 * then ; foo 999")
vm.do(": foo if if 1 2 + else 3 4 + then else if 5 6 + else 7 8 + then then ;")
vm.do("-1 -1 foo")
vm.do("0 -1 foo")
vm.do("-1 0 foo")
vm.do("0 0 foo")
vm.do("4 1 do i loop")
vm.do("3 1 do 40 10 do i j + 10 +loop loop")
vm.do("3 begin 1 - dup 0= until 999")
vm.do("4 begin 1 - dup 0= invert while 123 drop repeat 999")
vm.do(": foo 1 - if exit then 123 ;")
vm.do(": bar foo 999 ;")
vm.do("1 bar")
vm.do("2 bar")
vm.do(": foo 1 begin dup 1 + dup 5 = if exit then again ; foo")
vm.do("variable x 999 x ! 1 x +! x @")
vm.do("1 2 lshift")
vm.do("15 10 max")
vm.do("15 negate")
vm.do("true invert")
vm.do("false invert")
vm.do(": foo 10 5 do i exit loop ; 123 foo 456 789")


def myforth(source):
    vm = VirtualMachine()
    vm.do(source)
    return vm.stack.tolist()


def gforth(source):
    result = subprocess.run(
        'echo "100 maxdepth-.s ! {0} .s" | gforth'.format(source), shell=True, stdout=subprocess.PIPE
    ).stdout
    m = re.compile(rb"<[0-9]*>\s*((-?[0-9]+\s+)*)ok\s*$").search(result)
    if m is None:
        print(result.decode())
    return [int(x) for x in m.group(1).split()]


def testy(source):
    mine = myforth(source)
    theirs = gforth(source)
    assert mine == theirs, f"{source}\n\nmyforth: {mine}\n gforth: {theirs}"


testy(": foo 3 + 2 * ; 4 foo foo")
testy(": foo 3 + 2 * ; : bar 4 foo foo ; bar")
testy(": factorial dup 2 < if drop 1 exit then dup 1- recurse * ; 5 factorial")
testy("variable x 10 x ! 5 x +! x @ x @ x @")
testy("variable x 10 x ! 5 x +! : foo x @ x @ x @ ; foo foo")

testy(": foo if 999 then ; -1 foo")
testy(": foo if 999 then ; 0 foo")
testy(": foo if 999 then ; 1 foo")
testy(": foo if if 999 then then ; -1 -1 foo")
testy(": foo if if 999 then then ; 0 -1 foo")
testy(": foo if if 999 then then ; 1 -1 foo")
testy(": foo if if 999 then then ; -1 0 foo")
testy(": foo if if 999 then then ; 0 0 foo")
testy(": foo if if 999 then then ; 1 0 foo")
testy(": foo if if 999 then then ; -1 1 foo")
testy(": foo if if 999 then then ; 0 1 foo")
testy(": foo if if 999 then then ; 1 1 foo")
testy(": foo if 999 else 123 then ; -1 foo")
testy(": foo if 999 else 123 then ; 0 foo")
testy(": foo if 999 else 123 then ; 1 foo")
testy(": foo if 999 else 123 then ; -1 -1 foo")
testy(": foo if 999 else 123 then ; 0 -1 foo")
testy(": foo if 999 else 123 then ; 1 -1 foo")
testy(": foo if 999 else 123 then ; -1 0 foo")
testy(": foo if 999 else 123 then ; 0 0 foo")
testy(": foo if 999 else 123 then ; 1 0 foo")
testy(": foo if 999 else 123 then ; -1 1 foo")
testy(": foo if 999 else 123 then ; 0 1 foo")
testy(": foo if 999 else 123 then ; 1 1 foo")
testy(": foo if if 999 then else 123 then ; -1 -1 foo")
testy(": foo if if 999 then else 123 then ; 0 -1 foo")
testy(": foo if if 999 then else 123 then ; 1 -1 foo")
testy(": foo if if 999 then else 123 then ; -1 0 foo")
testy(": foo if if 999 then else 123 then ; 0 0 foo")
testy(": foo if if 999 then else 123 then ; 1 0 foo")
testy(": foo if if 999 then else 123 then ; -1 1 foo")
testy(": foo if if 999 then else 123 then ; 0 1 foo")
testy(": foo if if 999 then else 123 then ; 1 1 foo")
testy(": foo if 999 else if 123 then then ; -1 -1 foo")
testy(": foo if 999 else if 123 then then ; 0 -1 foo")
testy(": foo if 999 else if 123 then then ; 1 -1 foo")
testy(": foo if 999 else if 123 then then ; -1 0 foo")
testy(": foo if 999 else if 123 then then ; 0 0 foo")
testy(": foo if 999 else if 123 then then ; 1 0 foo")
testy(": foo if 999 else if 123 then then ; -1 1 foo")
testy(": foo if 999 else if 123 then then ; 0 1 foo")
testy(": foo if 999 else if 123 then then ; 1 1 foo")
testy(": foo if if 999 else 321 then else 123 then ; -1 -1 foo")
testy(": foo if if 999 else 321 then else 123 then ; 0 -1 foo")
testy(": foo if if 999 else 321 then else 123 then ; 1 -1 foo")
testy(": foo if if 999 else 321 then else 123 then ; -1 0 foo")
testy(": foo if if 999 else 321 then else 123 then ; 0 0 foo")
testy(": foo if if 999 else 321 then else 123 then ; 1 0 foo")
testy(": foo if if 999 else 321 then else 123 then ; -1 1 foo")
testy(": foo if if 999 else 321 then else 123 then ; 0 1 foo")
testy(": foo if if 999 else 321 then else 123 then ; 1 1 foo")
testy(": foo if 999 else if 123 else 321 then then ; -1 -1 foo")
testy(": foo if 999 else if 123 else 321 then then ; 0 -1 foo")
testy(": foo if 999 else if 123 else 321 then then ; 1 -1 foo")
testy(": foo if 999 else if 123 else 321 then then ; -1 0 foo")
testy(": foo if 999 else if 123 else 321 then then ; 0 0 foo")
testy(": foo if 999 else if 123 else 321 then then ; 1 0 foo")
testy(": foo if 999 else if 123 else 321 then then ; -1 1 foo")
testy(": foo if 999 else if 123 else 321 then then ; 0 1 foo")
testy(": foo if 999 else if 123 else 321 then then ; 1 1 foo")

testy(": foo do i loop ; 10 5 foo")
testy(": foo do i i +loop ; 100 5 foo")
testy(": foo 10 5 do 3 0 do 1+ loop loop ; 1 foo")
testy(": foo 10 5 do 3 0 do i loop loop ; foo")
testy(": foo 10 5 do 3 0 do j loop loop ; foo")
testy(": foo 10 5 do 3 0 do i j * loop loop ; foo")
testy(": foo 10 5 do 8 6 do 3 0 do i j * k * loop loop loop ; foo")

testy(": foo 3 begin dup 1 - dup 0= until ; foo")
testy(": foo 4 begin dup 1 - dup 0= invert while 123 drop repeat ; foo")
testy(": foo 3 begin dup 1 - dup 0= if exit then again ; foo")

testy("1 2 3 4 dup")
testy("1 2 3 4 drop")
testy("1 2 3 4 swap")
testy("1 2 3 4 over")
testy("1 2 3 4 rot")
testy("1 2 3 4 nip")
testy("1 2 3 4 tuck")

testy("3 5 +")
testy("-3 5 +")
testy("3 -5 +")
testy("-3 -5 +")
testy("3 5 -")
testy("-3 5 -")
testy("3 -5 -")
testy("-3 -5 -")
testy("5 3 -")
testy("5 -3 -")
testy("-5 3 -")
testy("-5 -3 -")
testy("3 5 *")
testy("-3 5 *")
testy("3 -5 *")
testy("-3 -5 *")
testy("22 7 /")
testy("-22 7 /")
testy("22 -7 /")
testy("-22 -7 /")
testy("22 7 mod")
testy("-22 7 mod")
testy("22 -7 mod")
testy("-22 -7 mod")
testy("22 7 /mod")
testy("-22 7 /mod")
testy("22 -7 /mod")
testy("-22 -7 /mod")
testy("0 1 lshift")
testy("0 2 lshift")
testy("0 3 lshift")
testy("1 1 lshift")
testy("1 2 lshift")
testy("1 3 lshift")
testy("5 1 lshift")
testy("5 2 lshift")
testy("5 3 lshift")
testy("-5 1 lshift")
testy("-5 2 lshift")
testy("-5 3 lshift")
testy("0 1 rshift")
testy("0 2 rshift")
testy("0 3 rshift")
testy("1 1 rshift")
testy("1 2 rshift")
testy("1 3 rshift")
testy("5 1 rshift")
testy("5 2 rshift")
testy("5 3 rshift")
testy("-5 1 rshift")
testy("-5 2 rshift")
testy("-5 3 rshift")
testy("-2 abs")
testy("-1 abs")
testy("0 abs")
testy("1 abs")
testy("2 abs")
testy("5 5 min")
testy("3 -5 min")
testy("-3 5 min")
testy("3 5 min")
testy("5 5 max")
testy("3 -5 max")
testy("-3 5 max")
testy("3 5 max")
testy("-2 negate")
testy("-1 negate")
testy("0 negate")
testy("1 negate")
testy("2 negate")
testy("-1 1+")
testy("0 1+")
testy("1 1+")
testy("-1 1-")
testy("0 1-")
testy("1 1-")
testy("-1 0=")
testy("0 0=")
testy("1 0=")
testy("5 5 =")
testy("3 -5 =")
testy("-3 5 =")
testy("3 5 =")
testy("5 5 <>")
testy("3 -5 <>")
testy("-3 5 <>")
testy("3 5 <>")
testy("5 5 >")
testy("3 -5 >")
testy("-3 5 >")
testy("3 5 >")
testy("5 5 >=")
testy("3 -5 >=")
testy("-3 5 >=")
testy("3 5 >=")
testy("5 5 <")
testy("3 -5 <")
testy("-3 5 <")
testy("3 5 <")
testy("5 5 <=")
testy("3 -5 <=")
testy("-3 5 <=")
testy("3 5 <=")
testy("-1 -1 and")
testy("0 -1 and")
testy("1 -1 and")
testy("-1 0 and")
testy("0 0 and")
testy("1 0 and")
testy("-1 1 and")
testy("0 1 and")
testy("1 1 and")
testy("-1 -1 or")
testy("0 -1 or")
testy("1 -1 or")
testy("-1 0 or")
testy("0 0 or")
testy("1 0 or")
testy("-1 1 or")
testy("0 1 or")
testy("1 1 or")
testy("-1 -1 xor")
testy("0 -1 xor")
testy("1 -1 xor")
testy("-1 0 xor")
testy("0 0 xor")
testy("1 0 xor")
testy("-1 1 xor")
testy("0 1 xor")
testy("1 1 xor")
testy("-1 invert")
testy("0 invert")
testy("1 invert")
testy("true")
testy("false")

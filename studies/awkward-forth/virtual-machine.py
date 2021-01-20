import math
import re
import subprocess
import sys

import numpy as np

import awkward as ak
import uproot
import skhep_testdata


class InputBuffer:
    def __init__(self, buffer):
        self.buffer = np.frombuffer(buffer, dtype=np.uint8)
        self.pointer = 0

    def __str__(self):
        return str(self.buffer)

    def __repr__(self):
        return "<InputBuffer {0}>".format(
            str(self).replace("\n", "\n                ")
        )

    def read(self, howmany, dtype):
        next = self.pointer + howmany * np.dtype(dtype).itemsize
        if next > len(self.buffer):
            raise ValueError("read beyond end of input buffer")
        out = self.buffer[self.pointer:next].view(dtype)
        self.pointer = next
        return out

    def seek(self, to):
        if not 0 <= to <= len(self.buffer):
            raise ValueError("seeked beyond end of input buffer")
        self.pointer = to

    def skip(self, howmany):
        next = self.pointer + howmany
        if not 0 <= next <= len(self.buffer):
            raise ValueError("skipped beyond end of input buffer")
        self.pointer = next

    @property
    def end(self):
        return self.pointer == len(self.buffer)

    @property
    def position(self):
        return self.pointer

    @property
    def length(self):
        return len(self.buffer)


class OutputBuffer:
    def __init__(self, dtype, initial=1024, resize=1.5):
        self.buffer = np.full(initial, 999, dtype=dtype)
        self.length = 0
        self.resize = resize

    def __str__(self):
        return str(self.buffer[:self.length])

    def __repr__(self):
        return "<OutputBuffer {0} ({1})>".format(
            str(self).replace("\n", "\n                "), str(self.buffer.dtype)
        )

    @property
    def dtype(self):
        return self.buffer.dtype

    def __array__(self):
        return self.buffer[:self.length]

    def tolist(self):
        return self.buffer[:self.length].tolist()

    def write(self, values):
        length = len(self.buffer)
        while self.length + len(values) > length:
            length = int(math.ceil(length * self.resize))

        if length > len(self.buffer):
            replacement = np.full(length, 999, dtype=self.buffer.dtype)
            replacement[:len(self.buffer)] = self.buffer
            self.buffer = replacement

        self.buffer[self.length : self.length + len(values)] = values
        self.length += len(values)

    def rewind(self, howmany):
        next = self.length - howmany
        if not 0 <= next:
            raise ValueError("rewinding before beginning of output")
        self.length = next


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
                            "output dtype must be one of: " + ", ".join(Builtin.dtypes)
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
                    if pointer + 1 < stop and tokenized[pointer + 1] == "seek":
                        instructions.append(Builtin.SEEK.as_integer)
                        instructions.append(input_names.index(word))
                        pointer += 2

                    elif pointer + 1 < stop and tokenized[pointer + 1] == "skip":
                        instructions.append(Builtin.SKIP.as_integer)
                        instructions.append(input_names.index(word))
                        pointer += 2

                    elif pointer + 1 < stop and tokenized[pointer + 1] == "end":
                        instructions.append(Builtin.END.as_integer)
                        instructions.append(input_names.index(word))
                        pointer += 2

                    elif pointer + 1 < stop and tokenized[pointer + 1] == "pos":
                        instructions.append(Builtin.POSITION.as_integer)
                        instructions.append(input_names.index(word))
                        pointer += 2

                    elif pointer + 1 < stop and tokenized[pointer + 1] == "len":
                        instructions.append(Builtin.LENGTH_INPUT.as_integer)
                        instructions.append(input_names.index(word))
                        pointer += 2

                    else:
                        if pointer + 1 >= stop or tokenized[pointer + 1] not in Builtin.ptypes:
                            raise ValueError(
                                "missing parser word after input name; must be one of: "
                                + ", ".join(repr(x) for x in Builtin.ptypes)
                                + " or 'seek', 'skip', 'end', 'pos', 'len'"
                            )
                        ptype = tokenized[pointer + 1]
                        pointer += 1

                        direct = False
                        output = None
                        if pointer + 1 < stop and tokenized[pointer + 1] in output_names:
                            direct = True
                            output = tokenized[pointer + 1]
                            pointer += 1

                        pointer += 1

                        instructions.append(Builtin.as_parser(ptype, direct))
                        instructions.append(input_names.index(word))
                        if direct:
                            instructions.append(output_names.index(output))

                elif word in output_names:
                    if pointer + 1 < stop and tokenized[pointer + 1] == "rewind":
                        instructions.append(Builtin.REWIND.as_integer)
                        instructions.append(output_names.index(word))
                        pointer += 2

                    elif pointer + 1 < stop and tokenized[pointer + 1] == "len":
                        instructions.append(Builtin.LENGTH_OUTPUT.as_integer)
                        instructions.append(output_names.index(word))
                        pointer += 2

                    else:
                        if pointer + 1 >= stop or tokenized[pointer + 1] != "<-":
                            raise ValueError(
                                "missing '<-' or 'rewind', 'len' word after output name"
                            )
                        pointer += 2

                        instructions.append(Builtin.WRITE.as_integer)
                        instructions.append(output_names.index(word))

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

    def run(
        self,
        instructions,
        variables,
        inputs,
        input_names=[],
        output_names=[],
        output_dtypes=[],
        variable_names=[],
        verbose=False
    ):
        # Create the stack.
        self.stack = Stack()

        # Ensure that the inputs are raw bytes.
        inputs = [InputBuffer(x) for x in inputs]

        # Create the outputs.
        outputs = [OutputBuffer(x) for x in output_dtypes]

        if verbose:
            print("{0:30.30s} | {1}".format("instruction", "stack before instruction"))
            print("-------------------------------+-------------------------")

            def printout(indent, instruction, showstack):
                if showstack:
                    showstack = str(self.stack)
                else:
                    showstack = ""
                print("{0:30.30s} | {1}".format("  " * indent + str(instruction), showstack))

        # Create a stack of instruction pointers to avoid native function calls.
        dictionary = [np.array(x, np.int32) for x in self.dictionary + [instructions]]
        which = [len(self.dictionary)]
        where = [0]

        # The do .. loop stack is a different stack.
        do_depth = []
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
                    do_stop.pop()
                    do_step.pop()
                    do_i.pop()
                    where[-1] += 1
                    continue

                else:
                    # Step forward in a DO loop.
                    if verbose:
                        printout(len(which) - 1, "do: {0}".format(do_i[-1]), False)

                if Builtin.is_parser(instruction):
                    inputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1

                    if verbose:
                        printout(
                            len(which) - 1,
                            input_names[inputnum] + " " + Builtin.from_parser(instruction),
                            True
                        )

                    howmany = 1
                    if Builtin.is_repeated(instruction):
                        howmany = self.stack.pop()

                    dtype = Builtin.dtype_of(instruction)
                    try:
                        values = inputs[inputnum].read(howmany, dtype)
                    except ValueError as err:
                        return str(err), outputs

                    if dtype not in (np.bool_, np.int8, np.uint8):
                        if bool(Builtin.is_bigendian(instruction)) ^ (sys.byteorder == "big"):
                            values = np.frombuffer(values, np.dtype(dtype).newbyteorder())

                    if Builtin.is_direct(instruction):
                        outputnum = dictionary[which[-1]][where[-1]]
                        where[-1] += 1
                        outputs[outputnum].write(values)
                    else:
                        for x in values:
                            self.stack.push(x)

                elif instruction == Builtin.LITERAL.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, num, True)
                    self.stack.push(num)

                elif instruction == Builtin.PUT.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, "! " + variable_names[num], True)
                    variables[num] = self.stack.pop()

                elif instruction == Builtin.INC.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, "+! " + variable_names[num], True)
                    variables[num] += self.stack.pop()

                elif instruction == Builtin.GET.as_integer:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, "@ " + variable_names[num], True)
                    self.stack.push(variables[num])

                elif instruction == Builtin.REWIND.as_integer:
                    outputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, output_names[outputnum] + " rewind", True)
                    try:
                        outputs[outputnum].rewind(self.stack.pop())
                    except ValueError as err:
                        return str(err), outputs

                elif instruction == Builtin.LENGTH_OUTPUT.as_integer:
                    outputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, output_names[outputnum] + " length", True)
                    self.stack.push(outputs[outputnum].length)

                elif instruction == Builtin.WRITE.as_integer:
                    outputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, output_names[outputnum] + " <-", True)
                    outputs[outputnum].write([self.stack.pop()])

                elif instruction == Builtin.SEEK.as_integer:
                    inputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, input_names[inputnum] + " seek", True)
                    try:
                        inputs[inputnum].seek(self.stack.pop())
                    except ValueError as err:
                        return str(err), outputs

                elif instruction == Builtin.SKIP.as_integer:
                    inputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, input_names[inputnum] + " skip", True)
                    try:
                        inputs[inputnum].skip(self.stack.pop())
                    except ValueError as err:
                        return str(err), outputs

                elif instruction == Builtin.END.as_integer:
                    inputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, input_names[inputnum] + " end", True)
                    self.stack.push(-1 if inputs[inputnum].end() else 0)

                elif instruction == Builtin.POSITION.as_integer:
                    inputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, input_names[inputnum] + " position", True)
                    self.stack.push(inputs[inputnum].position)

                elif instruction == Builtin.LENGTH_INPUT.as_integer:
                    inputnum = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, input_names[inputnum] + " length", True)
                    self.stack.push(inputs[inputnum].length)

                elif instruction == Builtin.IF.as_integer:
                    if verbose:
                        printout(len(which) - 1, "if", True)
                    if self.stack.pop() == 0:
                        # False, so then skip over the next instruction.
                        where[-1] += 1
                    else:
                        # True, so do the next instruction (i.e. normal flow).
                        if verbose:
                            printout(len(which) - 1, "then", False)

                elif instruction == Builtin.IF_ELSE.as_integer:
                    if verbose:
                        printout(len(which) - 1, "if", True)
                    if self.stack.pop() == 0:
                        if verbose:
                            printout(len(which) - 1, "else", False)
                        # False, so then skip over the next instruction but do the one after that.
                        where[-1] += 1
                    else:
                        if verbose:
                            printout(len(which) - 1, "then", False)
                        # True, so do the next instruction but skip the one after that.
                        consequent = dictionary[which[-1]][where[-1]]
                        where[-1] += 2
                        if verbose:
                            for name, value in self.dictionary_names.items():
                                if value == consequent:
                                    printout(len(which) - 1, name, False)
                                    break
                        which.append(consequent - len(Builtin.lookup))
                        where.append(0)

                elif instruction == Builtin.DO.as_integer:
                    do_depth.append(len(which))
                    do_start = self.stack.pop()
                    do_stop.append(self.stack.pop())
                    do_step.append(False)
                    do_i.append(do_start)

                elif instruction == Builtin.DO_STEP.as_integer:
                    do_depth.append(len(which))
                    do_start = self.stack.pop()
                    do_stop.append(self.stack.pop())
                    do_step.append(True)
                    do_i.append(do_start)

                elif instruction == Builtin.AGAIN.as_integer:
                    if verbose:
                        printout(len(which) - 1, "again", False)
                    # Go back and do the body again.
                    where[-1] -= 2

                elif instruction == Builtin.UNTIL.as_integer:
                    if verbose:
                        printout(len(which) - 1, "until", True)
                    if self.stack.pop() == 0:
                        # False, so go back and do the body again.
                        where[-1] -= 2

                elif instruction == Builtin.WHILE.as_integer:
                    if verbose:
                        printout(len(which) - 1, "while", True)
                    if self.stack.pop() == 0:
                        # False, so skip over the conditional body.
                        where[-1] += 1
                    else:
                        # True, so do the next instruction but skip back after that.
                        pretest = dictionary[which[-1]][where[-1]]
                        where[-1] -= 2
                        if verbose:
                            for name, value in self.dictionary_names.items():
                                if value == pretest:
                                    printout(len(which) - 1, name, False)
                                    break
                        which.append(pretest - len(Builtin.lookup))
                        where.append(0)

                elif instruction == Builtin.EXIT.as_integer:
                    if verbose:
                        printout(len(which) - 1, "exit", False)
                    exitdepth = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    for i in range(exitdepth):
                        which.pop()
                        where.pop()
                    while len(do_depth) != 0 and do_depth[-1] >= len(which):
                        do_depth.pop()
                        do_stop.pop()
                        do_step.pop()
                        do_i.pop()
                    break

                elif instruction == Builtin.I.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    if len(do_i) < 1:
                        self.stack.push(0)
                    else:
                        self.stack.push(do_i[-1])

                elif instruction == Builtin.J.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    if len(do_i) < 2:
                        self.stack.push(0)
                    else:
                        self.stack.push(do_i[-2])

                elif instruction == Builtin.K.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    if len(do_i) < 3:
                        self.stack.push(0)
                    else:
                        self.stack.push(do_i[-3])

                elif instruction == Builtin.DUP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    value = self.stack.pop()
                    self.stack.push(value)
                    self.stack.push(value)

                elif instruction == Builtin.DROP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.pop()

                elif instruction == Builtin.SWAP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(a)
                    self.stack.push(b)

                elif instruction == Builtin.OVER.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(b)
                    self.stack.push(a)
                    self.stack.push(b)

                elif instruction == Builtin.ROT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b, c = self.stack.pop(), self.stack.pop(), self.stack.pop()
                    self.stack.push(b)
                    self.stack.push(a)
                    self.stack.push(c)

                elif instruction == Builtin.NIP.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(a)

                elif instruction == Builtin.TUCK.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(a)
                    self.stack.push(b)
                    self.stack.push(a)

                elif instruction == Builtin.ADD.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() + self.stack.pop())

                elif instruction == Builtin.SUB.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(b - a)

                elif instruction == Builtin.MUL.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() * self.stack.pop())

                elif instruction == Builtin.DIV.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(b // a)

                elif instruction == Builtin.MOD.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(b % a)

                elif instruction == Builtin.DIVMOD.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    c, d = divmod(b, a)
                    self.stack.push(d)
                    self.stack.push(c)

                elif instruction == Builtin.LSHIFT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop(), self.stack.pop()
                    self.stack.push(b << a)

                elif instruction == Builtin.RSHIFT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    a, b = self.stack.pop().view(np.uint64), self.stack.pop().view(np.uint64)
                    self.stack.push(b >> a)

                elif instruction == Builtin.ABS.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(abs(self.stack.pop()))

                elif instruction == Builtin.MIN.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(min(self.stack.pop(), self.stack.pop()))

                elif instruction == Builtin.MAX.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(max(self.stack.pop(), self.stack.pop()))

                elif instruction == Builtin.NEGATE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-self.stack.pop())

                elif instruction == Builtin.ADD1.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() + 1)

                elif instruction == Builtin.SUB1.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() - 1)

                elif instruction == Builtin.EQ0.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() == 0 else 0)

                elif instruction == Builtin.EQ.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() == self.stack.pop() else 0)

                elif instruction == Builtin.NE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() != self.stack.pop() else 0)

                elif instruction == Builtin.GT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() < self.stack.pop() else 0)

                elif instruction == Builtin.GE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() <= self.stack.pop() else 0)

                elif instruction == Builtin.LT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() > self.stack.pop() else 0)

                elif instruction == Builtin.LE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1 if self.stack.pop() >= self.stack.pop() else 0)

                elif instruction == Builtin.AND.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() & self.stack.pop())

                elif instruction == Builtin.OR.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() | self.stack.pop())

                elif instruction == Builtin.XOR.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(self.stack.pop() ^ self.stack.pop())

                elif instruction == Builtin.INVERT.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(~self.stack.pop())

                elif instruction == Builtin.FALSE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(0)

                elif instruction == Builtin.TRUE.as_integer:
                    if verbose:
                        printout(len(which) - 1, Builtin.word(instruction), True)
                    self.stack.push(-1)

                elif instruction >= len(Builtin.lookup):
                    if verbose:
                        for name, value in self.dictionary_names.items():
                            if value == instruction:
                                printout(len(which) - 1, name, False)
                                break
                    which.append(instruction - len(Builtin.lookup))
                    where.append(0)

                else:
                    raise AssertionError(
                        "unrecognized machine code: {0}".format(instruction)
                    )

            which.pop()
            where.pop()

            if len(do_depth) > 0 and do_depth[-1] == len(which):
                if do_step[-1]:
                    do_i[-1] += self.stack.pop()
                else:
                    do_i[-1] += 1

        if verbose:
            print("{0:30.30s} | {1}".format("", str(self.stack)))

        return None, outputs

    def do(self, source, inputs={}, exceptions=False, verbose=False):
        if verbose:
            print("do: {0}".format(repr(source)))

        (
            instructions, variable_names, input_names, output_names, output_dtypes
        ) = self.compile(source)

        variables = [0 for x in variable_names]

        inputs = [inputs[x] for x in input_names]

        message, outputs = self.run(
            instructions,
            variables,
            inputs,
            input_names=input_names,
            output_names=output_names,
            output_dtypes=output_dtypes,
            variable_names=variable_names,
            verbose=verbose,
        )

        self.outputs = dict(zip(output_names, outputs))
        self.variables = dict(zip(variable_names, variables))

        if exceptions and message is not None:
            raise ValueError(message)
        else:
            return message

    def array(
        self,
        source,
        form,
        inputs={},
        length="length",
        key_format="{form_key}-{attribute}",
        exceptions=False,
        verbose=False,
    ):
        message = self.do(source, inputs=inputs, exceptions=exceptions, verbose=verbose)
        return ak.from_buffers(form, self.variables["length"], self.outputs, key_format=key_format)


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
            or word in ["skip", "seek", "end", "pos", "len", "rewind", "<-"]
            or word in Builtin.lookup
        )

    @staticmethod
    def is_parser(instruction):
        return bool(instruction & 0x80000000)

    @staticmethod
    def is_direct(instruction):
        return bool(instruction & 0x40000000)

    @staticmethod
    def is_repeated(instruction):
        return bool(instruction & 0x20000000)

    @staticmethod
    def is_bigendian(instruction):
        return bool(instruction & 0x10000000)

    @staticmethod
    def dtype_of(instruction):
        masked = instruction & 0x0fffffff
        if masked == Builtin.PARSER_BOOL:
            return np.bool_
        elif masked == Builtin.PARSER_INT8:
            return np.int8
        elif masked == Builtin.PARSER_INT16:
            return np.int16
        elif masked == Builtin.PARSER_INT32:
            return np.int32
        elif masked == Builtin.PARSER_INT64:
            return np.int64
        elif masked == Builtin.PARSER_SSIZE:
            return np.intp
        elif masked == Builtin.PARSER_UINT8:
            return np.uint8
        elif masked == Builtin.PARSER_UINT16:
            return np.uint16
        elif masked == Builtin.PARSER_UINT32:
            return np.uint32
        elif masked == Builtin.PARSER_UINT64:
            return np.uint64
        elif masked == Builtin.PARSER_USIZE:
            return np.uintp
        elif masked == Builtin.PARSER_FLOAT32:
            return np.float32
        elif masked == Builtin.PARSER_FLOAT64:
            return np.float64
        else:
            raise AssertionError(repr(masked))

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
            out |= Builtin.PARSER_BOOL
        elif ptype[0] == "b":
            out |= Builtin.PARSER_INT8
        elif ptype[0] == "h":
            out |= Builtin.PARSER_INT16
        elif ptype[0] == "i":
            out |= Builtin.PARSER_INT32
        elif ptype[0] == "q":
            out |= Builtin.PARSER_INT64
        elif ptype[0] == "n":
            out |= Builtin.PARSER_SSIZE
        elif ptype[0] == "B":
            out |= Builtin.PARSER_UINT8
        elif ptype[0] == "H":
            out |= Builtin.PARSER_UINT16
        elif ptype[0] == "I":
            out |= Builtin.PARSER_UINT32
        elif ptype[0] == "Q":
            out |= Builtin.PARSER_UINT64
        elif ptype[0] == "N":
            out |= Builtin.PARSER_USIZE
        elif ptype[0] == "f":
            out |= Builtin.PARSER_FLOAT32
        elif ptype[0] == "d":
            out |= Builtin.PARSER_FLOAT64
        else:
            raise AssertionError(ptype)
        return out

    @staticmethod
    def from_parser(instruction):
        n = ""
        if Builtin.is_repeated(instruction):
            n = "#"
        b = ""
        if Builtin.is_bigendian(instruction):
            b = "!"
        masked = instruction & 0x0fffffff
        if masked == Builtin.PARSER_BOOL:
            return n + b + "?"
        elif masked == Builtin.PARSER_INT8:
            return n + b + "b"
        elif masked == Builtin.PARSER_INT16:
            return n + b + "h"
        elif masked == Builtin.PARSER_INT32:
            return n + b + "i"
        elif masked == Builtin.PARSER_INT64:
            return n + b + "q"
        elif masked == Builtin.PARSER_SSIZE:
            return n + b + "n"
        elif masked == Builtin.PARSER_UINT8:
            return n + b + "B"
        elif masked == Builtin.PARSER_UINT16:
            return n + b + "H"
        elif masked == Builtin.PARSER_UINT32:
            return n + b + "I"
        elif masked == Builtin.PARSER_UINT64:
            return n + b + "Q"
        elif masked == Builtin.PARSER_USIZE:
            return n + b + "N"
        elif masked == Builtin.PARSER_FLOAT32:
            return n + b + "f"
        elif masked == Builtin.PARSER_FLOAT64:
            return n + b + "d"
        else:
            raise AssertionError(repr(masked))

Builtin.LITERAL = Builtin()
Builtin.PUT = Builtin()
Builtin.INC = Builtin()
Builtin.GET = Builtin()
Builtin.SKIP = Builtin()
Builtin.SEEK = Builtin()
Builtin.END = Builtin()
Builtin.POSITION = Builtin()
Builtin.LENGTH_INPUT = Builtin()
Builtin.REWIND = Builtin()
Builtin.LENGTH_OUTPUT = Builtin()
Builtin.WRITE = Builtin()
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


vector_of_vectors = uproot.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root"))["t/x"]
inputs = {
    "data": vector_of_vectors.basket(0).data,
    "byte_offsets": vector_of_vectors.basket(0).byte_offsets,
}
vm = VirtualMachine()
array = vm.array("""
input data
input byte_offsets
output outer-offsets int32
output inner-offsets int32
output content-data float64
variable length
variable outer-last
variable inner-last

0 length !
0 outer-last !
0 inner-last !
outer-last @ outer-offsets <-
inner-last @ inner-offsets <-

begin
  byte_offsets i->
  6 + data seek
  1 length +!

  data !i-> dup outer-last +!
  outer-last @ outer-offsets <-

  0 do
    data !i-> dup inner-last +!
    inner-last @ inner-offsets <-

    data #!d-> content-data
  loop
again
""",
      form={
          "class": "ListOffsetArray32",
          "offsets": "i32",
          "form_key": "outer",
          "content": {
              "class": "ListOffsetArray32",
              "offsets": "i32",
              "form_key": "inner",
              "content": {
                  "class": "NumpyArray",
                  "primitive": "float64",
                  "form_key": "content",
              },
          },
      },
      inputs=inputs)
assert array.tolist() == [
    [],
    [[], []],
    [[10.0], [], [10.0, 20.0]],
    [[20.0, -21.0, -22.0]],
    [[200.0], [-201.0], [202.0]],
]

vector_of_vectors = uproot.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root"))["t/x"]
inputs = {
    "data": vector_of_vectors.basket(0).data,
    "byte_offsets": vector_of_vectors.basket(0).byte_offsets,
}
vm = VirtualMachine()
vm.do("""
input data
input byte_offsets
begin
  byte_offsets pos byte_offsets len 4 - <
while
  byte_offsets i-> 6 + data seek data !i->
repeat
""", inputs=inputs)
assert vm.stack.tolist() == [0, 2, 3, 1, 3]

vm = VirtualMachine()
vm.do("""
output y float32
4 3 2 1
y <-
y <-
y <-
y <-
""")
assert vm.outputs["y"].dtype == np.dtype(np.float32)
assert vm.outputs["y"].tolist() == [1, 2, 3, 4]

vm = VirtualMachine()
vm.do("""
output y float32
4 3 2 1
y <-
y <-
1 y rewind
y <-
y <-
""")
assert vm.outputs["y"].dtype == np.dtype(np.float32)
assert vm.outputs["y"].tolist() == [1, 3, 4]

vm = VirtualMachine()
inputs = {"x": np.array([1, 2, 3, 4], np.int32)}
vm.do("""
input x
x i->
x i->
-8 x skip
x i->
x i->
x i->
x i->
""", inputs)
assert vm.stack.tolist() == [1, 2, 1, 2, 3, 4]

vm = VirtualMachine()
inputs = {"x": np.array([1, 2, 3, 4], np.int32)}
vm.do("""
input x
4 x #i->
""", inputs)
assert vm.stack.tolist() == [1, 2, 3, 4]

vm = VirtualMachine()
inputs = {"x": np.array([1, 2, 3, 4], np.int32)}
vm.do("""
input x
4 x #!i->
""", inputs)
assert vm.stack.tolist() == [16777216, 33554432, 50331648, 67108864]

vm = VirtualMachine()
inputs = {"x": np.array([1, 2, 3, 4], np.int32)}
vm.do("""
input x
8 x #h->
""", inputs)
assert vm.stack.tolist() == [1, 0, 2, 0, 3, 0, 4, 0]

vm = VirtualMachine()
inputs = {"x": np.array([1, 2, 3, 4], np.int32)}
vm.do("""
input x output y float32
x i-> y
x i-> y
x i-> y
x i-> y
""", inputs)
assert vm.outputs["y"].dtype == np.dtype(np.float32)
assert vm.outputs["y"].tolist() == [1, 2, 3, 4]

vm = VirtualMachine()
inputs = {"x": np.array([1, 2, 3, 4], np.int32)}
vm.do("""
input x output y float32
4 x #i-> y
""", inputs)
assert vm.outputs["y"].dtype == np.dtype(np.float32)
assert vm.outputs["y"].tolist() == [1, 2, 3, 4]

vm = VirtualMachine()
vm.do("3 ( whatever ) 2 + 2 *")
vm.do("3 ( whatever )\n2 + 2 *")
vm.do("3 \\ whatever\n2 + 2 *")
vm.do(": foo 3 2 + ; foo")
vm.do(": foo : bar 1 + ; 3 2 + ; foo bar")
vm.do("1 2 3 4 dup")
vm.do("1 2 3 4 drop")
vm.do("1 2 3 4 swap")
vm.do("1 2 3 4 over")
vm.do("1 2 3 4 rot")
vm.do("1 2 3 4 nip")
vm.do("1 2 3 4 tuck")
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
    print("""    vm32 = awkward.forth.ForthMachine32("{0}")
    vm32.run()
    assert vm32.stack == {1}
""".format(source, theirs))


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

testy(": foo 3 + 2 * ; 4 foo foo")
testy(": foo 3 + 2 * ; : bar 4 foo foo ; bar")
testy(": factorial dup 2 < if drop 1 exit then dup 1- recurse * ; 5 factorial")
testy("variable x 10 x ! 5 x +! x @ x @ x @")
testy("variable x 10 x ! 5 x +! : foo x @ x @ x @ ; foo foo")

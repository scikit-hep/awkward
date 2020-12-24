import math

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


builtin_instruction_names = [
    "LITERAL",
    "ADD",
]

class VirtualMachine:
    def __init__(self):
        for i, name in enumerate(builtin_instruction_names):
            setattr(self, name, i)
        self.num_builtin = len(builtin_instruction_names)

        # The dictionary contains user-defined functions as lists of instructions.
        self.dictionary_names = {}
        self.dictionary = []

    def compile(self, source):
        tokenized = source.split()

        def interpret(start, stop, instructions):
            pointer = start
            while pointer < stop:
                word = tokenized[pointer]

                if word == ":":
                    if pointer + 1 >= stop or tokenized[pointer + 1] == ";":
                        raise ValueError("missing name in definition")
                    name = tokenized[pointer + 1]

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

                    # Add the new word to the dictionary before interpreting it
                    # so that recursive functions can be defined.
                    self.dictionary_names[name] = self.num_builtin + len(self.dictionary)
                    self.dictionary.append([])

                    # Now interpret the subroutine and add it to the dictionary.
                    interpret(substart, substop, self.dictionary[-1])

                    pointer = substop + 1

                elif word == "+":
                    instructions.append(self.ADD)
                    pointer += 1

                else:
                    try:
                        num = int(word)
                    except ValueError:
                        try:
                            instruction = self.dictionary_names[word]
                        except KeyError:
                            raise ValueError("unrecognized word: " + repr(word))
                        else:
                            instructions.append(instruction)
                            pointer += 1
                    else:
                        instructions.append(self.LITERAL)
                        instructions.append(num)
                        pointer += 1

        instructions = []
        interpret(0, len(tokenized), instructions)
        return instructions

    def run(self, instructions, inputs, output_dtypes, verbose=False):
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
        which = [len(self.dictionary)]
        where = [0]
        dictionary = self.dictionary + [instructions]

        # Run through the instructions until the first stack layer reaches its end.
        while len(which) > 0:
            while where[-1] < len(dictionary[which[-1]]):
                instruction = dictionary[which[-1]][where[-1]]
                where[-1] += 1

                if instruction == self.LITERAL:
                    num = dictionary[which[-1]][where[-1]]
                    where[-1] += 1
                    if verbose:
                        printout(len(which) - 1, num, True)
                    stack.push(num)

                elif instruction == self.ADD:
                    if verbose:
                        printout(len(which) - 1, "+", True)
                    stack.push(stack.pop() + stack.pop())

                else:
                    if verbose:
                        for name, value in self.dictionary_names.items():
                            if value == instruction:
                                printout(len(which) - 1, name, False)
                                break
                    which.append(instruction - self.num_builtin)
                    where.append(0)

            which.pop()
            where.pop()

        if verbose:
            print("{0:20s} | {1}".format("", str(stack)))

        return outputs

    def do(self, source, inputs=[], output_dtypes=[], verbose=False):
        outputs = self.run(self.compile(source), inputs, output_dtypes, verbose=verbose)


vm = VirtualMachine()
vm.do("3 2 +", verbose=True)
vm.do(": foo 3 2 + ; foo", verbose=True)
vm.do(": foo : bar 1 + ; 3 2 + ; foo bar", verbose=True)
vm.do(": foo + foo ; 1 2 3 4 5 foo", verbose=True)

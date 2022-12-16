AwkwardForth virtual machines
-----------------------------

Introduction
============

AwkwardForth is a subset of `standard Forth <https://forth-standard.org/standard/words>`__ with some additional built-in words. It is a domain specific language for creating columnar Awkward Arrays from record-oriented data sources, especially for cases in which the deserialization procedure for the record-oriented data is not known until runtime. Typically, this is because the data has a type or schema that is discovered at runtime and that type determines how bytes of input are interpreted and in what order. This does not apply to columnar data sources, such as Apache Arrow, Parquet, or some ROOT data, such as numerical types (like ``int`` or ``float``) and jagged arrays of numbers (like ``std::vector<int>``). It does apply to record-oriented sources like ProtoBuf, Avro, and complex types in ROOT TTrees, such as ``std::vector<std::vector<int>>`` or unsplit classes. Note that ROOT's new RNTuple is entirely columnar.

The `Easy Forth <https://skilldrick.github.io/easyforth/>`__ one-page tutorial is an excellent introduction to the idea of Forth. In a nutshell, whereas functional programming strives for pure functions with no side effects, Forth operations consist purely of side effects: every operation changes the state of the machine, whether the global stack of integers, global variables, or in the case of AwkwardForth, positions in input buffers and data written to output buffers. It has almost no syntax, less even than Lisp, in that it consists entirely of whitespace-separated words interpreted in reverse Polish order. (Looping and branching constructs do have a recursive grammar, but they are exceptions.)

AwkwardForth is interpreted as a bytecode-compiled virtual machine. The source code is compiled into sequences of integer codes, one sequence per user-defined word or nested control flow (e.g. body of loops and conditional branches). This "compilation" is literal, like Python or Java bytecode—no optimization is attempted. It is interpreted by a virtual machine that (on my laptop) runs at about 5 ns per instruction. Instructions are 1‒3 bytecodes long, and each bytecode is a 32-bit integer (templated as ``I`` in C++, but only instantiated for ``int32_t``). For comparison, the Python virtual machine (on the same laptop) runs at about 900 ns per instruction (see `this comment <https://github.com/scikit-hep/awkward-1.0/pull/648#issuecomment-761296216>`__), so AwkwardForth is an "interpreter" in the same sense as CPython, but almost 200× faster, due to its specialization. Strictly mathematical calculations can be much faster in compiled, optimized C++, but strictly I/O operations (from RAM to RAM) is about the same, with C++ being only 1.8× faster in the limit of one 32-bit copy per instruction. If dozens or more bytes are copied per instruction, the gap between AwkwardForth and C++ becomes insignificant. Since AwkwardForth is intended for mostly I/O purposes, this is acceptable.

Forth's emphasis on state-changing operations would make it a terrible choice for vectorized accelerators like GPUs, but an FPGA implementation could be great: FPGAs have a much longer "compilation" time than even C++, so it would be advantageous for an FPGA to be configurable by Forth programs in the same sense as AwkwardForth. Such a thing could, for instance, read ROOT files directly from GHz Ethernet into machine learning models implemented with `hls4ml <https://fastmachinelearning.org/hls4ml/>`__.

Properties of the Awkward Array's ForthMachine
==============================================

This part of the documentation is the most in flux, since we'll likely add features to the ForthMachine to make debugging easier.

In C++, there are three classes:

- `ForthMachineOf<T, I> <https://awkward-array.org/doc/main/_static/doxygen/classawkward_1_1ForthMachineOf.html>`__, where ``T`` is the stack type (``int32_t`` or ``int64_t``) and ``I`` is the instruction type (only ``int32_t`` has been instantiated).
- `ForthInputBuffer <https://awkward-array.org/doc/main/_static/doxygen/classawkward_1_1ForthInputBuffer.html>`__ is an untyped input buffer, which wraps a ``std::shared_ptr<void>``. (Note that one operation, copying multiple numbers from the input buffer to the stack (not directly to output buffers), will temporarily mutate data in the buffer if they need to be byte-swapped. This is a temporary mutation, so the buffer can be used by other functions afterward, but not at the same time as the ForthMachine. This thread-unsafety could be changed in the future.)
- `ForthOutputBufferOf<OUT> <https://awkward-array.org/doc/main/_static/doxygen/classawkward_1_1ForthOutputBuffer.html>`__ is a typed output buffer, specialized by ``OUT``. (The fact that the write methods are virtual is not a performance bottleneck: putting the output type information into Forth bytecodes and using a ``switch`` statement to go to specialized method calls has identical performance for small copies and is up to 2× worse for large copies. C++ vtables are hard to beat.)

In Python, only the two instantiations of the ForthMachine are bound through pybind11:

.. code-block:: python

    >>> from awkward.forth import ForthMachine32
    >>> from awkward.forth import ForthMachine64

The methods available in Python are a subset of the ones in C++. (The fast, lookup-by-integer methods were omitted.)

A ForthMachine compiles its source code once when it is constructed; new code requires a new machine. This machine computes the sum of 3 and 5.

.. code-block:: python

    >>> vm = ForthMachine32("3 5 +")
    >>> vm.run()
    >>> vm.stack
    [8]

Controlling execution
*********************

A ForthMachine has 3 states: "not ready," "paused," and "done." There are 6 methods that control execution of a ForthMachine:

- ``run(inputs)``: resets the state of the machine, starting in any state, and runs the main code from the beginning. If control reaches a ``pause`` word, the machine goes into the "paused" state. Otherwise, it goes into the "done" state.
- ``begin(inputs)``: resets the state of the machine, starting in any state, and goes into a "paused" state before the first instruction in the main code.
- ``resume()``: starts execution from a "paused" state and continues until the end of the main code, resulting in "done," or until the end of a user-defined word, if a word was paused while being called (see below).
- ``call(word)``: starting from a "paused" or "done" state, executes a user-defined word. If this operation contains a ``pause`` word, the machine will need to be resumed (see above) to reach the end of the user-defined word. When the user-defined word is finished, the state of the machine will be "paused" or "done," depending on where it started.
- ``step()``: executes only one instruction, starting from a "pause" state, ending in a "pause" or "done" state, depending on whether the last instruction in the main code is reached. This only exists for debugging: normal pausing and resuming should be done with ``pause`` words and ``resume()`` calls.
- ``reset()``: resets the state of the machine and (unlike all of the above), clears the stack, all variables, and detaches the input and output buffers (which might be significant for cleaning up memory use).

Here are some examples of controlling the execution state of a ForthMachine.

Stepping through a program (for debugging only):

.. code-block:: python

    >>> vm = ForthMachine32("3 5 +")
    >>> vm.begin()
    >>> vm.stack
    []
    >>> vm.step()
    >>> vm.stack
    [3]
    >>> vm.step()
    >>> vm.stack
    [3, 5]
    >>> vm.step()
    >>> vm.stack
    [8]

Pausing and resuming execution:

.. code-block:: python

    >>> vm = ForthMachine32("1 2 pause 3 4")
    >>> vm.run()
    >>> vm.stack
    [1, 2]
    >>> vm.run()
    >>> vm.stack
    [1, 2]
    >>> vm.resume()
    >>> vm.stack
    [1, 2, 3, 4]

Halting execution:

.. code-block:: python

    >>> vm = ForthMachine32("1 2 halt 3 4")
    >>> vm.run()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 'user halt' in AwkwardForth runtime: user-defined error or stopping condition
    >>> vm.stack
    [1, 2]
    >>> vm.run(raise_user_halt=False)
    'user halt'
    >>> vm.stack
    [1, 2]
    >>> vm.resume()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 'not ready' in AwkwardForth runtime: call 'begin' before 'step' or 'resume' (note: check 'is_ready')

Calling a user-defined word:

.. code-block:: python

    >>> vm = ForthMachine32(": callme 1 2 3 4 ;")
    >>> vm.call("callme")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 'not ready' in AwkwardForth runtime: call 'begin' before 'step' or 'resume' (note: check 'is_ready')
    >>> vm.run()
    >>> vm.stack
    []
    >>> vm.call("callme")
    >>> vm.stack
    [1, 2, 3, 4]

Interaction between ``pause`` and calling a user-defined word:

.. code-block:: python

    >>> vm = ForthMachine32(": callme 123 pause 321 ; 1 2 pause 3 4")
    >>> vm.run()
    >>> vm.stack
    [1, 2]
    >>> vm.call("callme")
    >>> vm.stack
    [1, 2, 123]
    >>> vm.resume()
    >>> vm.stack
    [1, 2, 123, 321]
    >>> vm.resume()
    >>> vm.stack
    [1, 2, 123, 321, 3, 4]

Manipulating the stack outside of a program:

.. code-block:: python

    >>> vm = ForthMachine32("if 123 else 321 then")
    >>> vm.begin()
    >>> vm.stack
    []
    >>> vm.stack_push(-1)    # true
    >>> vm.stack
    [-1]
    >>> vm.resume()          # if pops the value and runs the first branch
    >>> vm.stack
    [123]
    >>> vm.begin()
    >>> vm.stack
    []
    >>> vm.stack_push(0)     # false
    >>> vm.stack
    [0]
    >>> vm.resume()          # if pops the value and runs the second branch
    >>> vm.stack
    [321]

Variables, inputs, and outputs
******************************

AwkwardForth can also have (global, scalar) variables, (global, untyped) inputs, and (global, typed) outputs. (The language has no nested scopes.) Here is an example of a ForthMachine with a variable:

.. code-block:: python

    >>> vm = ForthMachine32("variable x    10 x !")
    >>> vm["x"]
    0
    >>> vm.run()
    >>> vm["x"]
    10

Here is an example of a ForthMachine with an input (``i->`` reads data as a 4-byte integer and moves the position 4 bytes):

.. code-block:: python

    >>> import numpy as np
    >>> vm = ForthMachine32("input x    x i-> stack")
    >>> vm.run({"x": np.array([3, 2, 1], np.int32)})
    >>> vm.stack
    [3]
    >>> vm.input_position("x")
    4

Here is an example of a ForthMachine with an output (``<-`` writes data from the stack, converting it to the output type, if necessary):

.. code-block:: python

    >>> vm = ForthMachine32("output x int32    999    x <- stack")
    >>> vm.begin()
    >>> vm.step()
    >>> vm.stack
    [999]
    >>> np.asarray(vm["x"])
    array([], dtype=int32)
    >>> vm.step()
    >>> vm.stack
    []
    >>> np.asarray(vm["x"])
    array([999], dtype=int32)

(Note: always view outputs as NumPy arrays. In Awkward 1.x, outputs were returned as NumpyArray objects, but now they are returned as NumPy arrays. Wraping the output in ``np.asarray`` makes your code version-independent.)

A ForthMachine can have an arbitrary number of variables, inputs, and outputs, and an arbitrary number of user-defined words, with index orders defined by the order of declaration (relevant for fast C++ access).

AwkwardForth has no floating-point operations at all. (If we need to add one, it would be a separate floating-point stack, which is the typical way Forth implementations handle floating-point calculations, if at all.)

Inspecting the bytecode
***********************

The bytecode instructions for an AwkwardForth program are a ListOffsetArray of 32-bit integers, which can be inspected and decompiled.

.. code-block:: python

    >>> import awkward as ak
    >>> vm = ForthMachine32("if 123 else 321 then")
    >>> vm.bytecodes
    <ListOffsetArray len='3'>
        <offsets><Index dtype='int64' len='4'>[0 3 5 7]</Index></offsets>
        <content><NumpyArray dtype='int32' len='7'>[  4  60  61   0 123   0 321]</NumpyArray></content>
    </ListOffsetArray>
    >>> ak.Array(vm.bytecodes)
    <Array [[4, 60, 61], [0, 123], [0, 321]] type='3 * var * int32'>
    >>> print(vm.decompiled)
    if
      123
    else
      321
    then

Position in the code
********************

You can also get the current position in the bytecode (the position of the next instruction to be run) and a decompiled string of that instruction.

.. code-block:: python

    >>> vm = ForthMachine32("1 2 pause 3 4")
    >>> # Literal integers in the source code are two-bytecode instructions (0 followed by the number).
    >>> ak.Array(vm.bytecodes)
    <Array [[0, 1, 0, 2, 2, 0, 3, 0, 4]] type='1 * var * int32'>
    >>> vm.current_bytecode_position
    -1
    >>> vm.begin()
    >>> vm.current_bytecode_position
    0
    >>> vm.current_instruction
    '1'
    >>> vm.resume()
    >>> vm.current_bytecode_position
    5
    >>> vm.current_instruction
    '3'
    >>> vm.resume()
    >>> vm.current_bytecode_position
    -1
    >>> vm.current_instruction
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 'is done' in AwkwardForth runtime: reached the end of the program; call 'begin' to 'step' again (note: check 'is_done')

Note that this ``current_bytecode_position`` refers to the absolute position in ``bytecodes.content``, not a position relative to the beginning of a segment. The following example illustrates that, as well as the use of ``current_recursion_depth`` (PR `#653 <https://github.com/scikit-hep/awkward-1.0/pull/653>`__ may be required):

.. code-block:: python

    >>> vm = ForthMachine32("0 if 123 else 321 then")
    >>> ak.to_list(vm.bytecodes)
    [[0, 0, 4, 60, 61], [0, 123], [0, 321]]
    >>> vm.begin()
    >>> vm.current_bytecode_position, vm.current_recursion_depth, vm.current_instruction
    (0, 1, '0')
    >>> vm.step()
    >>> vm.current_bytecode_position, vm.current_recursion_depth, vm.current_instruction
    (2, 1, 'if\n  123\nelse\n  321\nthen')
    >>> vm.step()
    >>> vm.current_bytecode_position, vm.current_recursion_depth, vm.current_instruction
    (4, 1, '(anonymous segment at 2)')
    >>> vm.step()
    >>> vm.current_bytecode_position, vm.current_recursion_depth, vm.current_instruction
    (7, 2, '321')
    >>> vm.step()
    >>> vm.current_bytecode_position, vm.current_recursion_depth(-1, 1)

Performance counters
********************

As the ForthMachine executes code, it counts the number of instructions it encounters and the number of nanoseconds spent in the execution loop. This can be useful for quantifying algorithms.

.. code-block:: python

    >>> vm = ForthMachine32("5 3 + 2 *")
    >>> vm.count_instructions, vm.count_nanoseconds
    (0, 0)
    >>> vm.run()
    >>> vm.count_instructions, vm.count_nanoseconds
    (5, 6739)
    >>> vm.run()
    >>> vm.count_instructions, vm.count_nanoseconds
    (10, 15233)
    >>> vm.run()
    >>> vm.count_instructions, vm.count_nanoseconds
    (15, 23751)
    >>> vm.run()
    >>> vm.count_instructions, vm.count_nanoseconds
    (20, 32512)
    >>> vm.count_reset()
    >>> vm.count_instructions, vm.count_nanoseconds
    (0, 0)

In performance studies, keep in mind that only large samples are meaningful, since modern processors streamline code as it runs (moving data/Forth instructions from RAM into CPU cache, predicting branches, pipelining hardware instructions, etc.).

There are also counters for read instructions and write instructions.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y float64
    ... 
    ... 10 0 do
    ...   x d-> y
    ... loop
    ... """)
    >>> vm.run({"x": np.arange(10) * 1.1})
    >>> np.asarray(vm["y"])
    array([0. , 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    >>> vm.count_reads, vm.count_writes
    (10, 10)
    >>> vm.run({"x": np.arange(10) * 1.1})
    >>> vm.count_reads, vm.count_writes
    (20, 20)

Note that multi-read/write instructions (described below) count as one because they are much faster than individual read/writes.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y float64
    ... 
    ... 10 x #d-> y
    ... """)
    >>> vm.run({"x": np.arange(10) * 1.1})
    >>> np.asarray(vm["y"])
    array([0. , 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    >>> vm.count_reads, vm.count_writes
    (1, 1)
    >>> vm.run({"x": np.arange(10) * 1.1})
    >>> vm.count_reads, vm.count_writes
    (2, 2)

Also note that the execution ``reset()`` is independent of the performance-counter ``count_reset()``. Resetting one does not reset the other.

.. code-block:: python

    >>> vm.reset()
    >>> vm.count_instructions, vm.count_nanoseconds, vm.count_reads, vm.count_writes
    (4, 18769, 2, 2)
    >>> vm.count_reset()
    >>> vm.count_instructions, vm.count_nanoseconds, vm.count_reads, vm.count_writes
    (0, 0, 0, 0)

Documentation of standard words
===============================

`Comments <https://forth-standard.org/standard/core/p>`__
*********************************************************

Standard Forth has two types of comments: parentheses and backslash-to-end-of-line.

.. code-block:: python

    >>> vm = ForthMachine32("( This does nothing. )")
    >>> ak.Array(vm.bytecodes)
    <Array [[]] type='1 * var * int32'>
    >>> vm = ForthMachine32("1 2 ( comment ) 3 4")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 3, 4]
    >>> vm = ForthMachine32("""
    ... 1 2    \\ comment to end of line
    ... 3 4    \\ 2 backslashes in Python quotes -> 1 backslash in string
    ... """)
    >>> vm.run()
    >>> vm.stack
    [1, 2, 3, 4]

In both styles, you have to make sure that the "``(``", "``)``", and "``\``" characters are separated by a space; otherwise the tokenizer won't recognize them as distinct from another word. (That is, "``(comment)``" is not "``( comment )``".) Also, parentheses are closed by the first _balancing_ close-parenthesis.

.. code-block:: python

    >>> vm = ForthMachine32("( outer ( inner ) still a comment )")
    >>> ak.Array(vm.bytecodes)
    <Array [[]] type='1 * var * int32'>

Literal integers
****************

Literal integers in the source code put an integer on the stack. AwkwardForth has no floating point types, so only ``-?[0-9]+`` are allowed, no ``.`` or ``e``. If the number is prefixed by ``0x``, then the number is parsed as hexidecimal, with ``-?[0-9a-f]`` allowed.

.. code-block:: python

    >>> vm = ForthMachine32("1 2 -3 04 0xff")
    >>> vm.run()
    >>> vm.stack
    [1, 2, -3, 4, 255]

Constant strings
****************

Some syntactical elements, such as ``enum`` and ``enumonly`` use constant strings as part of their syntax. Strings in Forth begin with a ``s"`` token, separated by one space from the beginning of the string itself, and continue to an unescaped closing quote, ``"``, which is not separated by whitespace from the end of the string.

An escaped quote is preceded by a slash, like ``\"`` (which may be ``\\"`` or ``\\\"`` in Python strings, depending on how they, themselves, are quoted). Escaped quotes do not close a string, only an unescaped quote.

Here are some examples. Strings can be extracted from the source code using ``string_at``; they are numbered by their appearance in the source code, starting at zero.

.. code-block:: python

    >>> source_code = r's" simple" s" two words" s" nested \"quotes\"" s"   extra space   "'
    >>> print(source_code)
    s" simple" s" two words" s" nested \"quotes\"" s"   extra space   "
    >>> vm = ForthMachine32(source_code)
    >>> print(vm.string_at(0))
    simple
    >>> print(vm.string_at(1))
    two words
    >>> print(vm.string_at(2))
    nested "quotes"
    >>> print(vm.string_at(3))
      extra space   
    >>> vm.string_at(3)
    '  extra space   '

(The escapes didn't have to be escaped in the Python string above because we used ``r'`` to start the Python string. If you need to use ``'`` or ``"``, the quoting behavior will be different. It's a good idea to print out source code with strings to be sure you're giving Forth what you think you are.)

If strings are used in source code outside of a syntactic element, their behavior is to push two items onto the stack: a string address and the string length. String addresses in AwkwardForth are the sequential identifiers, starting with zero, that can be passed to ``string_at``.

Continuing with the above example,

.. code-block:: python

    >>> vm.run()
    >>> vm.stack
    [0, 6, 1, 9, 2, 15, 3, 16]

because

- "``simple``" is string ``0`` with length ``6``,
- "``two words``" is string ``1`` with length ``9``,
- "``nested "quotes"``" is string ``2`` with length ``15``,
- the string with extra spaces is string ``3`` with length ``16``.

Constant strings were only include in AwkwardForth for use in syntactic elements and their runtime behavior is implemented to be consistent with Standard Forth, but it's not very useful. (Standard Forth uses pointer positions for string identifiers and has words for manipulating bytes in memory, which makes strings useful but dangerous.)

Printing to standard output for debugging
*****************************************

Forth has several words for printing to standard output, and they're included in AwkwardForth for debugging.

- ``."`` starts quoting a string exactly like ``s"``, but it prints that string to standard output. This is good for short messages to trace the control flow through the program. It does not print a carriage return unless the string contains one.
- ``.`` pops an integer from the stack and prints it. If you want to peek at the top of the stack without modifying it, write ``dup .``. It does not print a carriage return.
- ``.s`` prints the whole stack without modifying it. It does not print a carriage return.
- ``cr`` prints a carriage return; it usually follows one of the above to make the output easier to read.

Example:

.. code-block:: python

    >>> vm = ForthMachine32('0 1 2 3 ." almost there" cr 4 5 dup . cr .s cr')
    >>> vm.run()
    almost there
    5 
    <6> 0 1 2 3 4 5 <- top 
    >>> vm.stack
    [0, 1, 2, 3, 4, 5]

The number in angle brackets (``<`` and ``>``) is the size of the stack and stack print-outs always end with "``<- top``".

User defined words: `: .. ; <https://forth-standard.org/standard/core/Colon>`__
*******************************************************************************

The main distinction between Forth and a stack-based assembly language is that Forth allows the programmer to define new words. These words are like subroutines, but do not have formal argument lists or return values: they manipulate the stack like any built-in word. A word's "informal" arguments are the items it pops off the stack when it begins and its "informal" return values are the items it pushes onto the stack when it ends.

It is customary to document a word with a comment like

.. code-block:: forth

    : sum-of-squares ( x y -- sum )
      dup *          ( x y -- x y*y )
      swap           ( x y*y -- y*y x )
      dup *          ( y*y x -- y*y x*x )
      +              ( sum )
    ;

That is, the state of the top of the stack (the rightmost end is the "top," where items get pushed and popped) before the operation is to the left of two hyphens "``--``" and the state of the top of the stack afterward is to the right. Here is that example as a ForthMachine:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... : sum-of-squares ( x y -- sum )
    ...   dup *          ( x y -- x y*y )
    ...   swap           ( x y*y -- y*y x )
    ...   dup *          ( y*y x -- y*y x*x )
    ...   +              ( sum )
    ... ;
    ... 3 4 sum-of-squares
    ... """)
    >>> vm.run()
    >>> vm.stack
    [25]

User-defined words are used like any other word—in reverse Polish order. Thus, ``3 4 sum-of-squares`` calls this newly defined word.

In AwkwardForth, words can be defined after they are used, and they can call themselves by name recursively. (Not all Forths allow that.) All declarations (new words, variables, inputs, and outputs) are compiled in a global namespace when a ForthMachine is constructed. However, words can only call previously defined words or themselves because this compilation proceeds in one pass. (It's also possible to define a word inside of a definition of a word, but there is no value in doing so, because namespaces are not scoped and Forth has no notion of a closure.)

Note that a "common error" is to forget a space between the colon ("``:``") and the word it defines or the semicolon ("``;``") and the last word in the definition.

`recurse <https://forth-standard.org/standard/core/RECURSE>`__
**************************************************************

AwkwardForth functions can call themselves for recursion, but the standard defines ``recurse`` to allow it in systems without this ability. It is included for convenience in porting examples from other Forths. For example, Fibonacci numbers from `this page <http://cubbi.com/fibonacci/forth.html>`__:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... : fibonacci    ( n -- nth-fibonacci-number )
    ...   dup
    ...   1 > if
    ...     1- dup 1- recurse
    ...     swap recurse
    ...     +
    ...   then
    ... ;
    ... 20 0 do
    ...   i fibonacci
    ... loop
    ... """)
    >>> vm.run()
    >>> vm.stack
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]

(In this example, the word ``recurse`` could be replaced with ``fibonacci`` because AwkwardForth's function table includes the function currently being defined.)

`if .. then <https://forth-standard.org/standard/core/IF>`__
************************************************************

The ``if .. then`` brackets a sequence of words, pops one value of the stack, does nothing if that value is exactly ``0``, and does the bracketed words if it is non-zero. Conventionally, ``-1`` is used as "true" because it is the bitwise inversion of ``0`` (in `two's complement arithmetic <https://en.wikipedia.org/wiki/Two%27s_complement>`__).

Note that the word "``then``" acts as a _terminator_ of the code branch: it comes _after_ the code to run if the predicate is true. This is just a weird rule to remember.

.. code-block:: python

    >>> vm = ForthMachine32("if 1 2 3 4 then")
    >>> vm.begin()
    >>> vm.stack_push(0)
    >>> vm.resume()
    >>> vm.stack
    []
    >>> vm.begin()
    >>> vm.stack_push(-1)
    >>> vm.resume()
    >>> vm.stack
    [1, 2, 3, 4]

`if .. else .. then <https://forth-standard.org/standard/core/ELSE>`__
**********************************************************************

The ``if .. else .. then`` brackets two sequences of words, pops one value off the stack, does the first if that value is non-zero and the second if that value is zero.

.. code-block:: python

    >>> vm = ForthMachine32("if 123 else 321 then")
    >>> vm.begin()
    >>> vm.stack_push(0)
    >>> vm.resume()
    >>> vm.stack
    [321]
    >>> vm.begin()
    >>> vm.stack_push(-1)
    >>> vm.resume()
    >>> vm.stack
    [123]

`case .. of .. endof .. endcase <https://forth-standard.org/standard/core/CASE>`__
**********************************************************************************

The ``case .. of .. endof .. endcase`` structure is an extension of ``if .. else .. then`` that allows a single expression to be matched against many possible values. It's Forth's equivalent of C's ``switch`` statement.

This complex expression evaluates in the following order:

.. code-block:: forth

    expression-to-pop case
      value-to-compare-1 of consequent-to-evaluate-1 endof
      value-to-compare-2 of consequent-to-evaluate-2 endof
      value-to-compare-3 of consequent-to-evaluate-3 endof
                            optional-default-to-evaluate
    endcase

That is, the ``case`` word pops a value off the stack and compares it with the expressions before each ``of``. The first one that matches invokes the corresponding consequent to evaluate, which is nested between ``of`` and ``endof``. A default expression to evaluate, if none of the values match, comes after all of the ``of``-``endof`` pairs but before the ``endcase``. The ``endcase`` closes the block.

If all of the values to compare are literal integers, then this structure compiles to a table-lookup. If not, then it compiles to the equivalent ``if .. else .. then`` chain. The reason this structure was added to AwkwardForth was to take advantage of this optimization.

Here is a ``case .. of .. endof .. endcase`` that compiles to a table-lookup. The single item that it consumes from the stack is passed in from outside the machine.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... case
    ...   1 of ." one" cr endof
    ...   2 of ." two" cr endof
    ...   3 of ." three" cr endof
    ...        ." something else" cr
    ... endcase
    ... """)
    >>> vm.begin()
    >>> vm.stack_push(0)
    >>> vm.resume()
    something else
    >>> vm.begin()
    >>> vm.stack_push(1)
    >>> vm.resume()
    one
    >>> vm.begin()
    >>> vm.stack_push(2)
    >>> vm.resume()
    two
    >>> vm.begin()
    >>> vm.stack_push(3)
    >>> vm.resume()
    three
    >>> vm.begin()
    >>> vm.stack_push(4)
    >>> vm.resume()
    something else

The above is a table-lookup because all items to the left of each ``of`` is a literal integer. It still works if one of them is a runtime expression, but not as fast.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... case
    ...   1     of ." one" cr endof
    ...   1 1 + of ." two" cr endof
    ...   3     of ." three" cr endof
    ...            ." something else" cr
    ... endcase
    ... """)
    >>> vm.begin()
    >>> vm.stack_push(1)
    >>> vm.resume()
    one
    >>> vm.begin()
    >>> vm.stack_push(2)
    >>> vm.resume()
    two
    >>> vm.begin()
    >>> vm.stack_push(3)
    >>> vm.resume()
    three

The ``case .. of .. endof .. endcase`` construct can be used to match type codes in an input stream, in which different values precede different behavior, or even string constants (like "``{``" or "``[``" in JSON) via ``enum`` or ``enumonly``.

`do .. loop <https://forth-standard.org/standard/core/DO>`__
************************************************************

The ``do .. loop`` brackets a sequence of words, pops two values off the stack, "stop" and "start," and repeats the bracketed sequence "stop minus start" times. Note that the top of the stack is the starting value and the second-to-top is the stopping value, so they read backward. Here are two examples:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... 10 0 do
    ...   123
    ... loop
    ... """)
    >>> vm.run()
    >>> vm.stack
    [123, 123, 123, 123, 123, 123, 123, 123, 123, 123]

As described below, ``i`` is the current state of the incrementing variable.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... 10 0 do
    ...   i
    ... loop
    ... """)
    >>> vm.run()
    >>> vm.stack
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Although the "start" and "stop" values may be constants in the code, they are pulled from the stack, so they can be determined at runtime.

`do .. +loop <https://forth-standard.org/standard/core/PlusLOOP>`__
*******************************************************************

The ``do .. +loop`` brackets a sequence of words, pops two values off the stack, "stop" and "start," and repeats the bracketed sequence. At the end of the bracketed sequence, another value is popped off the stack, "step", which indicates how much the incrementing variable changes in each step.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... 100 0 do
    ...   i
    ...   10
    ... +loop
    ... """)
    >>> vm.run()
    >>> vm.stack
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

Like "start" and "stop," the "step" value is pulled from the stack, so it can be determined at runtime.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... 1000 1 do
    ...   i
    ...   dup 2 *
    ... +loop
    ... """)
    >>> vm.run()
    >>> vm.stack
    [1, 3, 9, 27, 81, 243, 729]

`i, j, and k <https://forth-standard.org/standard/core/I>`__
************************************************************

The letters ``i``, ``j``, and ``k`` are reserved words whose values are set by ``do`` loops and nested ``do`` loops (up to three levels).

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... 10 5 do
    ...   8 3 do
    ...     5 0 do
    ...       k 100 * j 10 * i + +
    ...     loop
    ...   loop
    ... loop
    ... """)
    >>> vm.run()
    >>> vm.stack
    [530, 531, 532, 533, 534,
     540, 541, 542, 543, 544,
     550, 551, 552, 553, 554,
     560, 561, 562, 563, 564,
     570, 571, 572, 573, 574,

     630, 631, 632, 633, 634,
     640, 641, 642, 643, 644,
     650, 651, 652, 653, 654,
     660, 661, 662, 663, 664,
     670, 671, 672, 673, 674,

     730, 731, 732, 733, 734,
     740, 741, 742, 743, 744,
     750, 751, 752, 753, 754,
     760, 761, 762, 763, 764,
     770, 771, 772, 773, 774,

     830, 831, 832, 833, 834,
     840, 841, 842, 843, 844,
     850, 851, 852, 853, 854,
     860, 861, 862, 863, 864,
     870, 871, 872, 873, 874,

     930, 931, 932, 933, 934,
     940, 941, 942, 943, 944,
     950, 951, 952, 953, 954,
     960, 961, 962, 963, 964,
     970, 971, 972, 973, 974]

`begin .. again <https://forth-standard.org/standard/core/AGAIN>`__
*******************************************************************

The ``begin .. again`` brackets a sequence of words and repeats them indefinitely. Only an error or a control-flow construct like ``exit``, ``halt``, and ``pause`` can break out of it. Programs can be simplified by repeating indefinitely and ignoring errors.

.. code-block:: python

    >>> vm = ForthMachine32("input x begin x i-> stack again")
    >>> vm.run({"x": np.arange(10, dtype=np.int32)}, raise_read_beyond=False)
    'read beyond'
    >>> vm.stack
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

`begin .. until <https://forth-standard.org/standard/core/UNTIL>`__
*******************************************************************

The ``begin .. until`` brackets a sequence of words and repeats them, popping a value from the stack at the end of the sequence, and using that value to determine whether to continue. If the value is ``0``, the body repeats; otherwise, it stops. This is a posttest loop: the condition is part of the repeated body.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... 10
    ... begin
    ...   dup 1-
    ...   dup 0=
    ... until
    ... """)
    >>> vm.run()
    >>> vm.stack
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

`begin .. while .. repeat <https://forth-standard.org/standard/core/WHILE>`__
*****************************************************************************

The ``begin .. while .. repeat`` brackets two sequences of words, executes the first unconditionally and, if non-zero, executes the second sequence. At the end of the second sequence, the control returns to the first sequence to re-evaluate the condition. This is a pretest loop: the condition has to be separated from the loop body like the parenthesized condition in a ``while`` loop in C:

.. code-block:: c

    while (condition) {
      body
    }

`exit <https://forth-standard.org/standard/core/EXIT>`__
********************************************************

The ``exit`` word provides a non-local return from a word.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... : recursive   ( n -- n n-1 )
    ...   dup 0= if
    ...     exit
    ...   then
    ...   dup 1-
    ...   recursive
    ... ;
    ... 10 recursive
    ... """)
    >>> vm.run()
    >>> vm.stack
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

If you're familiar with other Forths, note that AwkwardForth does not need an `unloop <https://forth-standard.org/standard/core/UNLOOP>`__ to clean up after incomplete ``do .. loop`` constructs.

`Variable declaration <https://forth-standard.org/standard/core/VARIABLE>`__
****************************************************************************

Variables are declared with ``variable`` followed by a name.

.. code-block:: python

    >>> vm = ForthMachine32("variable x")
    >>> vm["x"]
    0

Variables have the same numerical type as the stack and global scope.

(In Forth code, you should try to use the stack instead of named variables.)

`Variable !, +!, and @ <https://forth-standard.org/standard/core/Store>`__
**************************************************************************

A variable name followed by "``!``" pops a value from the stack and assigns it to the variable.

A variable name followed by "``+!``" pops a value from the stack and adds it to the variable.

A variable name followed by "``@``" pushes the value of the variable to the stack.


.. code-block:: python

    >>> vm = ForthMachine32("""
    ... variable x
    ... 10 x !
    ... 5 x +!
    ... x @
    ... """)
    >>> vm.run()
    >>> vm.stack
    [15]

`dup <https://forth-standard.org/standard/core/DUP>`__, `drop <https://forth-standard.org/standard/core/DROP>`__, `swap <https://forth-standard.org/standard/core/SWAP>`__, `over <https://forth-standard.org/standard/core/OVER>`__, `rot <https://forth-standard.org/standard/core/ROT>`__, `nip <https://forth-standard.org/standard/core/NIP>`__, `tuck <https://forth-standard.org/standard/core/TUCK>`__
**************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

These are the standard stack manipulation words.

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 dup")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 3, 4, 4]

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 drop")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 3]

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 swap")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 4, 3]

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 over")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 3, 4, 3]

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 rot")
    >>> vm.run()
    >>> vm.stack
    [1, 3, 4, 2]

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 nip")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 4]

.. code-block:: python

    >>> vm = ForthMachine32("1 2 3 4 tuck")
    >>> vm.run()
    >>> vm.stack
    [1, 2, 4, 3, 4]

`+ <https://forth-standard.org/standard/core/Plus>`__, `- <https://forth-standard.org/standard/core/Minus>`__, `* <https://forth-standard.org/standard/core/Times>`__, `/ <https://forth-standard.org/standard/core/Div>`__, `mod <https://forth-standard.org/standard/core/MOD>`__, `/mod <https://forth-standard.org/standard/core/DivMOD>`__
***********************************************************************************************************************************************************************************************************************************************************************************************************************************************

Four-function arithmetic. For asymmetric operations (subtraction, division, and modulo), note the order of arguments: second-to-top first, then top.

.. code-block:: python

    >>> vm = ForthMachine32("3 5 +")
    >>> vm.run()
    >>> vm.stack
    [8]

.. code-block:: python

    >>> vm = ForthMachine32("3 5 -")
    >>> vm.run()
    >>> vm.stack
    [-2]

.. code-block:: python

    >>> vm = ForthMachine32("3 5 *")
    >>> vm.run()
    >>> vm.stack
    [15]

Forth, like Python and unlike C and Java, performs floor division, rather than integer division, so negative values round toward minus infinity, rather than rounding toward zero.

.. code-block:: python

    >>> vm = ForthMachine32("22 7 /")
    >>> vm.run()
    >>> vm.stack
    [3]
    >>> vm = ForthMachine32("-22 7 /")
    >>> vm.run()
    >>> vm.stack
    [-4]

Forth, like Python and unlike C and Java, performs modulo, rather than remainder, so negative values round toward minus infinity, rather than rounding toward zero.

.. code-block:: python

    >>> vm = ForthMachine32("22 7 mod")
    >>> vm.run()
    >>> vm.stack
    [1]
    >>> vm = ForthMachine32("-22 7 mod")
    >>> vm.run()
    >>> vm.stack
    [6]

The ``/mod`` operation does division and modulo in a single instruction. It pushes two values onto the stack.

.. code-block:: python

    >>> vm = ForthMachine32("22 7 /mod")
    >>> vm.run()
    >>> vm.stack
    [1, 3]

Division by zero is one of the possible error states for a ForthMachine.

.. code-block:: python

    >>> vm = ForthMachine32("22 0 /")
    >>> vm.run()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 'division by zero' in AwkwardForth runtime: tried to divide by zero

`negate <https://forth-standard.org/standard/core/NEGATE>`__, `1+ <https://forth-standard.org/standard/core/OnePlus>`__, `1- <https://forth-standard.org/standard/core/OneMinus>`__, `abs <https://forth-standard.org/standard/core/ABS>`__
*******************************************************************************************************************************************************************************************************************************************

Unary functions pop one value from the stack and push the result.

.. code-block:: python

    >>> vm = ForthMachine32("12 negate")
    >>> vm.run()
    >>> vm.stack
    [-12]

.. code-block:: python

    >>> vm = ForthMachine32("12 1+")
    >>> vm.run()
    >>> vm.stack
    [13]

.. code-block:: python

    >>> vm = ForthMachine32("12 1-")
    >>> vm.run()
    >>> vm.stack
    [11]

.. code-block:: python

    >>> vm = ForthMachine32("-12 abs")
    >>> vm.run()
    >>> vm.stack
    [12]

`min <https://forth-standard.org/standard/core/MIN>`__ and `max <https://forth-standard.org/standard/core/MAX>`__
*****************************************************************************************************************

The ``min`` and ``max`` words pop two values from the stack and push one.

.. code-block:: python

    >>> vm = ForthMachine32("3 5 min")
    >>> vm.run()
    >>> vm.stack
    [3]
    >>> vm = ForthMachine32("3 5 max")
    >>> vm.run()
    >>> vm.stack
    [5]

`= <https://forth-standard.org/standard/core/Equal>`__, `<> <https://forth-standard.org/standard/core/ne>`__, `> <https://forth-standard.org/standard/core/more>`__, >=, `< <https://forth-standard.org/standard/core/less>`__, <=
**********************************************************************************************************************************************************************************************************************************

Comparison operators pop two values from the stack and either push ``-1`` (true) or ``0`` (false).

Note that equality is a single "``=``" and inequality is "``<>``".

Standard Forth does not have greater-or-equal or less-or-equal, but they are the obvious extensions.

`0= <https://forth-standard.org/standard/core/ZeroEqual>`__
***********************************************************

The ``0=`` word checks for equality with zero, which is useful for normalizing booleans to ``0`` and ``-1``.

`invert <https://forth-standard.org/standard/core/INVERT>`__, `and <https://forth-standard.org/standard/core/AND>`__, `or <https://forth-standard.org/standard/core/OR>`__, `xor <https://forth-standard.org/standard/core/XOR>`__
**********************************************************************************************************************************************************************************************************************************

Instead of logical operators, Forth has bitwise operators. For ``invert`` to serve as logical-not, the non-zero value must be ``-1``, so normalize it with ``0=``.

.. code-block:: python

    >>> vm = ForthMachine32("0 invert")
    >>> vm.run()
    >>> vm.stack
    [-1]
    >>> vm = ForthMachine32("-1 invert")
    >>> vm.run()
    >>> vm.stack
    [0]
    >>> vm = ForthMachine32("1 invert")
    >>> vm.run()
    >>> vm.stack
    [-2]

Likewise, ``and`` and ``or`` are bitwise-and and bitwise-or.

.. code-block:: python

    >>> vm = ForthMachine32("1 2 or")
    >>> vm.run()
    >>> vm.stack
    [3]
    >>> vm = ForthMachine32("1 2 and")
    >>> vm.run()
    >>> vm.stack
    [0]

`lshift <https://forth-standard.org/standard/core/LSHIFT>`__ and `rshift <https://forth-standard.org/standard/core/RSHIFT>`__
*****************************************************************************************************************************

Left bitwise-shift and right bitwise-shift are good for bit fiddling.

`false <https://forth-standard.org/standard/core/FALSE>`__ and `true <https://forth-standard.org/standard/core/TRUE>`__
***********************************************************************************************************************

The ``false`` and ``true`` words are useful mnemonics for ``0`` and ``-1``. They make source code easier to read.

Documentation of built-in words specialized for I/O
===================================================

AwkwardForth's input and output handling words are not standard Forth, but a reasonable extension of it for this domain-specific purpose.

Input declaration
*****************

Input buffers are declared in the same way as variables. If an input has been declared in the source code, it must be provided in the ForthMachine's ``run(inputs)`` and ``begin(inputs)`` methods.

.. code-block:: python

    >>> vm = ForthMachine32("input x")
    >>> vm.run()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: AwkwardForth source code defines an input that was not provided: x
    >>> import numpy as np
    >>> vm.run({"x": np.array([1, 2, 3])})

Input read
**********

All of the words that read from an input buffer have the form: "``input-name *-> output-name``". The "``input-name``" is the name of one of the declared input buffers, the "``output-name``" is either a declared output buffer name or the special word "``stack``", and the "``*->``" is a word that ends in "``->``". There are 46 different words that end in "``->``". They are described below.

To an output buffer or to the stack
"""""""""""""""""""""""""""""""""""

The destination for a read operation can either be an output buffer or the stack. Directly reading from input to output is faster and more information-preserving than reading from input to the stack and then writing from the stack to the output.

Here's an example of reading directly to an output buffer:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y float64
    ... 
    ... x d-> y
    ... x d-> y
    ... x d-> y
    ... """)
    >>> vm.run({"x": np.array([1.1, 2.2, 3.3])})
    >>> vm["y"]
    array([1.1, 2.2, 3.3])

Here is an example that goes through the stack:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y float64
    ... 
    ... x d-> stack   y <- stack
    ... x d-> stack   y <- stack
    ... x d-> stack   y <- stack
    ... """)
    >>> vm.run({"x": np.array([1.1, 2.2, 3.3])})
    >>> np.asarray(vm["y"])
    array([1., 2., 3.])

Since the stack of this ForthMachine32 consists of 32-bit integers, the floating-point inputs get truncated before they can be written to the floating-point output.

You'd only want to copy inputs to the stack before copying them to the output if you need to manipulate them in some way, and the only manipulations relevant in parsing are integer operations, such as cumulative sums and identifying seek points.

Single value vs multiple values
"""""""""""""""""""""""""""""""

Reading a batch of data in one instruction is faster than reading the same data in many steps. To read a batch of data, prepend the "``*->``" word with a number sign (``#``). This pops a value off the stack to use as the number of items to read.

The following examples result in the same output:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y float32
    ... 
    ... 1000000 0 do
    ...   x d-> y
    ... loop
    ... """)
    >>> vm.run({"x": np.arange(1000000) * 1.1})
    >>> np.asarray(vm["y"])
    array([0.0000000e+00, 1.1000000e+00, 2.2000000e+00, ..., 1.0999968e+06,
           1.0999978e+06, 1.0999989e+06], dtype=float32)

and

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y float32
    ... 
    ... 1000000 x #d-> y
    ... """)
    >>> vm.run({"x": np.arange(1000000) * 1.1})
    >>> np.asarray(vm["y"])
    array([0.0000000e+00, 1.1000000e+00, 2.2000000e+00, ..., 1.0999968e+06,
           1.0999978e+06, 1.0999989e+06], dtype=float32)

but the second is faster because it involves two Forth instructions and one ``memcpy``.

Type codes
""""""""""

Inputs are untyped; their interpretation depends on the sequence of Forth commands. The letter immediately preceding the "``->``" specifies this interpretation—those letters were taken from `Python's struct module <https://docs.python.org/3/library/struct.html#format-characters>`__. The format-letters recognized by AwkwardForth are:

- ``?`` for ``bool``: 1 byte, false if exactly zero, true if nonzero;
- ``b`` for ``int8``: 1-byte signed integer;
- ``h`` for ``int16``: 2-byte signed integer;
- ``i`` for ``int32``: 4-byte signed integer;
- ``q`` for ``int64``: 8-byte signed integer;
- ``n`` for platform-dependent ``ssize_t``: 4 or 8 bytes, signed integer;
- ``B`` for ``int8``: 1-byte unsigned integer;
- ``H`` for ``int16``: 2-byte unsigned integer;
- ``I`` for ``int32``: 4-byte unsigned integer;
- ``Q`` for ``int64``: 8-byte unsigned integer;
- ``N`` for platform-dependent ``ssize_t``: 4 or 8 bytes, unsigned integer;
- ``f`` for ``float32``: 4-byte floating-point number;
- ``d`` for ``float64``: 8-byte floating-point number.

Since each read increments the input position, the choice of format also affects the resulting position in the file.

Here is an example of reading ``int32`` values as though they were ``int16``:

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... 
    ... 10 0 do
    ...   x h-> stack
    ... loop
    ... """)
    >>> vm.run({"x": np.arange(5, dtype=np.int32)})
    >>> vm.stack
    [0, 0, 1, 0, 2, 0, 3, 0, 4, 0]

Here is the same thing with an ``int32`` output. They are still interpreted as ``int16`` because the read command is ``h->``, even though they are then converted to a ``int32`` output.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y int32
    ... 
    ... 10 0 do
    ...   x h-> y
    ... loop
    ... """)
    >>> vm.run({"x": np.arange(5, dtype=np.int32)})
    >>> np.asarray(vm["y"])
    array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0], dtype=int32)

Big-endian vs little-endian
"""""""""""""""""""""""""""

The formatters in the previous section all assume the data are little-endian (regardless of the architecture for which Awkward Array is compiled). To read big-endian values, the formatter must be preceded by a "``!``".

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... 
    ... 10 0 do
    ...   x !i-> stack
    ... loop
    ... """)
    >>> vm.run({"x": np.arange(10, dtype=np.int32)})
    >>> vm.stack
    [0, 16777216, 33554432, 50331648, 67108864, 83886080, 100663296, 117440512, 134217728, 150994944]

The two modifiers, "``#``" and "``!``", must be in order: "``#``" first.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... 
    ... 10 x #!i-> stack
    ... """)
    >>> vm.run({"x": np.arange(10, dtype=np.int32)})
    >>> vm.stack
    [0, 16777216, 33554432, 50331648, 67108864, 83886080, 100663296, 117440512, 134217728, 150994944]

Special type codes
******************

In addition to the type codes that can be expressed by combining "``#``", "``!``", and a character from `Python's struct module <https://docs.python.org/3/library/struct.html#format-characters>`__, there are a few special codes for multi-byte sequences that occur in data formats fairly often.

These special type codes can be modified by "``#``" (with the same meaning as above), but most cannot be modified by "``!``" (all exceptions are noted).

Variable-length integers
""""""""""""""""""""""""

``varint->`` interprets the input as `variable-length unsigned integers <https://lucene.apache.org/java/3_5_0/fileformats.html#VInt>`__, in which the high bit indicates whether the next byte of input data is to be included in the calculation of this integer.

**Examples:**

- ``b"\x00"`` → ``0``
- ``b"\x01"`` → ``1``
- ``b"\x7f"`` → ``127``
- ``b"\x80\x01"`` → ``128``
- ``b"\x81\x01"`` → ``129``

Numbers less than ``2**7`` are encoded in 1 byte, other numbers less than ``2**14`` are encoded in 2 bytes, other numbers less than ``2**21`` are encoded in 3 bytes, etc.

``zigzag->`` interprets the input as `zig-zag variable-length signed integers <https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints>`__, which is like the above except that the unsigned ``n`` computed from a variable-length encoding is mapped to ``(n >> 1) ^ (-(n & 1))``, which are signed integers that alternate with increasing distance from zero.

**Examples:**

- ``b"\x00"`` → ``0``
- ``b"\x01"`` → ``-1``
- ``b"\x02"`` → ``1``
- ``b"\x03"`` → ``-2``
- ``b"\x04"`` → ``2``

The closer a number is to zero (with either sign), the fewer bytes its encoding has.

Unusual-length integers
"""""""""""""""""""""""

``2bit->``, ``3bit->``, ``4bit->``, etc. for any number of bits (up to 64). This type code can be modified by "``!``". Interprets the input as unsigned integers of an arbitrary number of bytes, and if repeated with "``#``", those bits can be packed, such that boundaries between numbers don't end on byte boundaries. At the end of the sequence, however, the last full byte of input is consumed.

**Example:** In the following, a sequence of 8 increasing 3-bit unsigned integers is encoded in a Python number using a binary literal, ``0b_000_001_010_011_100_101_110_111``. Since 24-bit numbers are not supported by NumPy, it is loaded into an 32-bit unsigned integer (4 bytes). The Forth expression ``8 x #3bit-> y`` interprets the 8 3-bit numbers and writes them to the output. The sequence is backward because NumPy created the buffer in this example on a little-endian machine (x86). After interpreting these 8 3-bit numbers, the input position has advanced 24 bits, or 3 bytes, not the whole 4 bytes available in the input.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... output y int32
    ... 
    ... ." begin: " x pos . cr   \\ show the starting input position
    ... 
    ... 8 x #3bit-> y
    ... 
    ... ." end:   " x pos . cr   \\ show the position after reading 8 3-bit integers
    ... ." total: " x len . cr   \\ show the total length of the input
    ... """)
    >>> vm.run({"x": np.array([0b_000_001_010_011_100_101_110_111], np.uint32)})
    begin: 0 
    end:   3 
    total: 4 
    >>> np.asarray(vm["y"])
    array([7, 6, 5, 4, 3, 2, 1, 0], dtype=int32)

Numbers and strings from ASCII text
"""""""""""""""""""""""""""""""""""

``textint->`` interprets the input as a signed integer, written in ASCII text.

**Examples:**

- ``b"123"`` → ``123``
- ``b"-999"`` → ``-999``

``textfloat->`` interprets the input as a floating-point number, written in ASCII text.

**Example:**

- ``b"-3.14e5"`` → ``-3.14 × 10⁵``.

Valid `JSON <https://www.json.org/>`__ numbers are accepted. Floating point numbers sent to the stack are truncated: send this directly to a floating-point output to avoid data loss.

``quotedstr->`` interprets the input as a quoted string, which must be sent to a ``uint8`` output. The interpretation starts with a quote character, ``b"\x22"``, and continues until it reaches an unescaped quote character. Valid `JSON <https://www.json.org/>`__ string escapes are accepted.

Recognizing strings or constant byte-patterns
*********************************************

Some data formats have reserved words or other constant byte-patterns with special meaning. Forth operates with a stack of integers, so the ``enum`` and ``enumonly`` words convert them into integers for decision-making (e.g. with ``case .. of .. endof .. endcase``).

The ``enum`` and ``enumonly`` words have the same syntax: they must follow an input name and are each followed by at least one string (opened with ``s"`` and closed by a word that ends with ``"``). These strings are the possible values to look for in the input. At runtime, they consume as many bytes from the input as are in the matching string and push one value onto the stack. If none of the strings after ``enum`` match, it pushes ``-1`` onto the stack. If none of the strings after ``enumonly`` match, it raises an exception.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... 
    ... 5 0 do
    ...   x skipws
    ...   x enum s" zero" s" one" s" two" s" three"
    ... loop
    ... """)
    >>> vm.run({"x": b"  zero  three two one four  "})
    >>> vm.stack
    [0, 3, 2, 1, -1]

If we replace ``enum`` with ``enumonly``, the last token ("``four``") raises an exception.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... 
    ... 5 0 do
    ...   x skipws
    ...   x enumonly s" zero" s" one" s" two" s" three"
    ... loop
    ... """)
    >>> vm.run({"x": b"  zero  three two one four  "})
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 'enumeration missing' in AwkwardForth runtime: expected one of several enumerated values in the input text, didn't find one

Peeking at a byte
*****************

The ``peek`` word copies the next input byte onto the stack, but it does not move the input position. Syntactically, it must follow an input name.

Input seek, skip, skipws
************************

The following words pop a value off the stack and use it to move the input buffer's position without reading:

- ``seek``: jumps to an absolute position within the file;
- ``skip``: moves a relative number of bytes in the file.

Since input buffers are untyped, absolute and relative positions are expressed in number of bytes.

The following word only moves an input buffer past any whitespace, as defined in `JSON <https://www.json.org/>`__ (space ``b" "``, linefeed ``b"\n"``, carriage return ``b"\r"``, or horizontal tab ``b"\t"``). It does not push or pop anything on the stack.

- ``skipws``: moves past any whitespace in the file.

Syntactically, all of the above must follow an input name.

Input len, pos, end
*******************

The following words can be written after an input name to push information about the input onto the stack:

- ``len``: length of the input (does not change);
- ``pos``: position in the input (changes with every read, ``seek``, and ``skip``);
- ``end``: true (``-1``) if the position is at the end of the input buffer; false (``0``) otherwise.

Since input buffers are untyped, lengths and positions are expressed in number of bytes.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... input x
    ... 
    ... 10 0 do
    ...   x i-> stack
    ...   drop
    ... loop
    ... 
    ... x len
    ... x pos
    ... x end
    ... """)
    >>> vm.run({"x": np.arange(10, dtype=np.int32)})
    >>> vm.stack
    [40, 40, -1]

Output declaration
******************

Output buffers are declared like input buffers, but with a type.

Whereas inputs must be provided as an argument to the ``run(input)`` and ``begin(input)`` methods, the outputs are produced by the ForthMachine and can be retrieved through ``__getitem__``.

.. code-block:: python

    >>> vm = ForthMachine32("output x float64")
    >>> vm.begin()
    >>> np.asarray(vm["x"])
    array([], dtype=float64)

Output types
""""""""""""

The following are allowed output buffer types:

- ``bool``: 1-byte booleans;
- ``int8``: 1-byte signed integers;
- ``int16``: 2-byte signed integers;
- ``int32``: 4-byte signed integers;
- ``int64``: 8-byte signed integers;
- ``uint8``: 1-byte unsigned integers;
- ``uint16``: 2-byte unsigned integers;
- ``uint32``: 4-byte unsigned integers;
- ``uint64``: 8-byte unsigned integers;
- ``float32``: 4-byte floating-point numbers;
- ``float64``: 8-byte floating-point numbers.

Output write
************

In some cases, outputs can be directly written from the inputs (see above). This is the fastest case. If the data need manipulation before they can be written, they have to come from the stack; the word for that is ``<- stack``.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... output x int32
    ... 
    ... 1 2 3 4
    ... x <- stack
    ... x <- stack
    ... x <- stack
    ... x <- stack
    ... """)
    >>> vm.run()
    >>> np.asarray(vm["x"])
    array([4, 3, 2, 1], dtype=int32)

Output add-and-write
********************

For outputs that represent cumulative sums (such as all ListOffsetArray outputs), replacing ``<- stack`` with ``+<- stack`` writes the sum of the last output value and the stack value, rather than just the stack value.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... output x int32
    ... 
    ... 100 5 5 5
    ... x +<- stack
    ... x +<- stack
    ... x +<- stack
    ... x +<- stack
    ... """)
    >>> vm.run()
    >>> np.asarray(vm["x"])
    array([  5,  10,  15, 115], dtype=int32)

The alternative to this would be to maintain a running sum in a variable associated with each output, or somehow manage to keep the running sum at the right level of the stack.

Output dup
**********

For outputs that need to repeat a specified number of times, an output name followed by ``dup`` duplicates the last value in the output. The number of times is a number popped from the stack.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... output x int32
    ... 
    ... 123 x <- stack
    ... 10 x dup
    ... """)
    >>> vm.run()
    >>> np.asarray(vm["x"])
    array([123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123], dtype=int32)

Output len
**********

If an output name is followed by ``len``, it pushes the _current_ length of the output onto the stack. This length is measured in the number of items, not the number of bytes.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... output x int32
    ... 
    ... x len
    ... 10 0 do
    ...   123 x <- stack
    ... loop
    ... x len
    ... """)
    >>> vm.run()
    >>> vm.stack
    [0, 10]

Output rewind
*************

If an output name is followed by ``rewind``, it pops a positive value off the stack and moves that many items backward, effectively erasing written data.

.. code-block:: python

    >>> vm = ForthMachine32("""
    ... output x int32
    ... 
    ... x len
    ... 10 0 do
    ...   123 x <- stack
    ... loop
    ... x len
    ... 3 x rewind
    ... x len
    ... """)
    >>> vm.run()
    >>> np.asarray(vm["x"])
    array([123, 123, 123, 123, 123, 123, 123], dtype=int32)
    >>> vm.stack
    [0, 10, 7]

Documentation of built-in words for control flow
================================================

The following words are not in Standard Forth. They exist to control the ForthMachine. Both of them can be used like any other word (e.g. they can appear in conditional branches or user-defined words).

halt
****

The ``halt`` word puts the ForthMachine into a "done" state, no matter where it is in execution. It also raises the "user halt" error, which can be silenced using ``raise_user_halt=False``.

.. code-block:: python

    >>> vm = ForthMachine32("halt")
    >>> vm.run(raise_user_halt=False)
    'user halt'

pause
*****

The ``pause`` word stops execution of the ForthMachine in such a way that execution can continue by calling ``resume()`` on the machine.

.. code-block:: python

    >>> vm = ForthMachine32("1 2 pause 3 4")
    >>> vm.run()
    >>> vm.stack
    [1, 2]
    >>> vm.resume()
    >>> vm.stack
    [1, 2, 3, 4]

Pausing is described in greater detail above.

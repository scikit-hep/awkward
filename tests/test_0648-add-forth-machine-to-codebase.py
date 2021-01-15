# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth


# def test_basics():
#     vm32 = awkward.forth.ForthMachine32("")
#     vm64 = awkward.forth.ForthMachine64("")

#     assert str(ak.type(vm32.bytecodes)) == "var * int32"
#     assert str(ak.type(vm64.bytecodes)) == "var * int32"
#     assert vm32.dictionary == []
#     assert vm64.dictionary == []
#     assert vm32.stack_max_depth == 1024
#     assert vm64.stack_max_depth == 1024
#     assert vm32.recursion_max_depth == 1024
#     assert vm64.recursion_max_depth == 1024
#     assert vm32.output_initial_size == 1024
#     assert vm64.output_initial_size == 1024
#     assert vm32.output_resize_factor == 1.5
#     assert vm64.output_resize_factor == 1.5

#     vm32.stack_push(1)
#     vm32.stack_push(2)
#     vm32.stack_push(3)
#     assert vm32.stack_pop() == 3
#     assert vm32.stack == [1, 2]
#     vm32.stack_clear()
#     assert vm32.stack == []
#     vm64.stack_push(1)
#     vm64.stack_push(2)
#     vm64.stack_push(3)
#     assert vm64.stack_pop() == 3
#     assert vm64.stack == [1, 2]
#     vm64.stack_clear()
#     assert vm64.stack == []

#     assert vm32.variables == {}
#     assert vm64.variables == {}
#     assert vm32.outputs == {}
#     assert vm64.outputs == {}


# def test_comment_compilation():
#     vm32 = awkward.forth.ForthMachine32("")
#     assert ak.to_list(vm32.bytecodes) == [[]]
#     assert vm32.decompiled == ""

#     vm32 = awkward.forth.ForthMachine32("( comment )")
#     assert ak.to_list(vm32.bytecodes) == [[]]

#     vm32 = awkward.forth.ForthMachine32("\\ comment")
#     assert ak.to_list(vm32.bytecodes) == [[]]

#     vm32 = awkward.forth.ForthMachine32("\\ comment\n")
#     assert ak.to_list(vm32.bytecodes) == [[]]

#     vm32 = awkward.forth.ForthMachine32("1 2 ( comment ) 3 4")
#     assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]

#     vm32 = awkward.forth.ForthMachine32("1 2 \\ comment \n 3 4")
#     assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]


# def test_literal_compilation():
#     vm32 = awkward.forth.ForthMachine32("1 2 3 4")
#     assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]
#     assert (
#         vm32.decompiled
#         == """1
# 2
# 3
# 4
# """
#     )


# def test_userdef_compilation():
#     vm32 = awkward.forth.ForthMachine32(": stuff 1 2 3 4 ;")
#     assert ak.to_list(vm32.bytecodes) == [[], [0, 1, 0, 2, 0, 3, 0, 4]]
#     assert (
#         vm32.decompiled
#         == """: stuff
#   1
#   2
#   3
#   4
# ;
# """
#     )

#     vm32 = awkward.forth.ForthMachine32(": foo 123 : bar 1 2 3 ; 321 ;")
#     assert ak.to_list(vm32.bytecodes) == [[], [0, 123, 0, 321], [0, 1, 0, 2, 0, 3]]
#     assert (
#         vm32.decompiled
#         == """: foo
#   123
#   321
# ;

# : bar
#   1
#   2
#   3
# ;
# """
#     )

#     vm32 = awkward.forth.ForthMachine32(": empty ;")
#     assert ak.to_list(vm32.bytecodes) == [[], []]
#     assert (
#         vm32.decompiled
#         == """: empty
# ;
# """
#     )

#     vm32 = awkward.forth.ForthMachine32(": infinite recurse ;")
#     assert ak.to_list(vm32.bytecodes) == [[], [60]]
#     assert (
#         vm32.decompiled
#         == """: infinite
#   infinite
# ;
# """
#     )


# def test_declarations_compilation():
#     vm32 = awkward.forth.ForthMachine32("variable x")
#     assert ak.to_list(vm32.bytecodes) == [[]]
#     assert (
#         vm32.decompiled
#         == """variable x
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("variable x variable y")
#     assert ak.to_list(vm32.bytecodes) == [[]]
#     assert (
#         vm32.decompiled
#         == """variable x
# variable y
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x")
#     assert ak.to_list(vm32.bytecodes) == [[]]
#     assert (
#         vm32.decompiled
#         == """input x
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("output x int32")
#     assert ak.to_list(vm32.bytecodes) == [[]]
#     assert (
#         vm32.decompiled
#         == """output x int32
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("output x int32 input y variable z")
#     assert ak.to_list(vm32.bytecodes) == [[]]
#     assert (
#         vm32.decompiled
#         == """variable z
# input y
# output x int32
# """
#     )

#     for dtype in [
#         "bool",
#         "int8",
#         "int16",
#         "int32",
#         "int64",
#         "uint8",
#         "uint16",
#         "uint32",
#         "uint64",
#         "float32",
#         "float64",
#     ]:
#         vm32 = awkward.forth.ForthMachine32("output x " + dtype)
#         assert vm32.decompiled == "output x " + dtype + "\n"


# def test_control_compilation():
#     vm32 = awkward.forth.ForthMachine32("halt")
#     assert (
#         vm32.decompiled
#         == """halt
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("pause")
#     assert (
#         vm32.decompiled
#         == """pause
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("exit")
#     assert (
#         vm32.decompiled
#         == """exit
# """
#     )


# def test_if_compilation():
#     vm32 = awkward.forth.ForthMachine32("-1 if then")
#     assert (
#         vm32.decompiled
#         == """-1
# if
# then
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("-1 if 3 5 + then")
#     assert (
#         vm32.decompiled
#         == """-1
# if
#   3
#   5
#   +
# then
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("-1 if else then")
#     assert (
#         vm32.decompiled
#         == """-1
# if
# else
# then
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("-1 if 3 5 + else 123 then")
#     assert (
#         vm32.decompiled
#         == """-1
# if
#   3
#   5
#   +
# else
#   123
# then
# """
#     )

#     vm32 = awkward.forth.ForthMachine32(": foo -1 if 3 5 + then ; foo")
#     assert (
#         vm32.decompiled
#         == """: foo
#   -1
#   if
#     3
#     5
#     +
#   then
# ;

# foo
# """
#     )


# def test_loop_compilation():
#     vm32 = awkward.forth.ForthMachine32("10 0 do loop")
#     assert (
#         vm32.decompiled
#         == """10
# 0
# do
# loop
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("10 0 do i loop")
#     assert (
#         vm32.decompiled
#         == """10
# 0
# do
#   i
# loop
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("10 0 do i +loop")
#     assert (
#         vm32.decompiled
#         == """10
# 0
# do
#   i
# +loop
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("10 0 do 5 0 do 3 1 do i j k loop loop loop")
#     assert (
#         vm32.decompiled
#         == """10
# 0
# do
#   5
#   0
#   do
#     3
#     1
#     do
#       i
#       j
#       k
#     loop
#   loop
# loop
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("begin again")
#     assert (
#         vm32.decompiled
#         == """begin
# again
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("begin 123 again")
#     assert (
#         vm32.decompiled
#         == """begin
#   123
# again
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("begin until")
#     assert (
#         vm32.decompiled
#         == """begin
# until
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("begin 123 until")
#     assert (
#         vm32.decompiled
#         == """begin
#   123
# until
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("begin while repeat")
#     print(ak.to_list(vm32.bytecodes))
#     assert (
#         vm32.decompiled
#         == """begin
# while
# repeat
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("begin 123 while 321 repeat")
#     print(ak.to_list(vm32.bytecodes))
#     assert (
#         vm32.decompiled
#         == """begin
#   123
# while
#   321
# repeat
# """
#     )


# def test_io_compilation():
#     vm32 = awkward.forth.ForthMachine32("variable x x !")
#     assert (
#         vm32.decompiled
#         == """variable x

# x !
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("variable x x +!")
#     assert (
#         vm32.decompiled
#         == """variable x

# x +!
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("variable x x @")
#     assert (
#         vm32.decompiled
#         == """variable x

# x @
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x x len")
#     assert (
#         vm32.decompiled
#         == """input x

# x len
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x x pos")
#     assert (
#         vm32.decompiled
#         == """input x

# x pos
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x x end")
#     assert (
#         vm32.decompiled
#         == """input x

# x end
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x x seek")
#     assert (
#         vm32.decompiled
#         == """input x

# x seek
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x x skip")
#     assert (
#         vm32.decompiled
#         == """input x

# x skip
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("output x int32 x len")
#     assert (
#         vm32.decompiled
#         == """output x int32

# x len
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("output x int32 x rewind")
#     assert (
#         vm32.decompiled
#         == """output x int32

# x rewind
# """
#     )


# def test_read_compilation():
#     vm32 = awkward.forth.ForthMachine32("input x x i-> stack")
#     assert (
#         vm32.decompiled
#         == """input x

# x i-> stack
# """
#     )

#     vm32 = awkward.forth.ForthMachine32("input x output y int32 x i-> y")
#     print(ak.to_list(vm32.bytecodes))
#     print(vm32.decompiled)
#     assert (
#         vm32.decompiled
#         == """input x
# output y int32

# x i-> y
# """
#     )

#     for rep in ["", "#"]:
#         for big in ["", "!"]:
#             for tpe in [
#                 "?",
#                 "b",
#                 "h",
#                 "i",
#                 "q",
#                 "n",
#                 "B",
#                 "H",
#                 "I",
#                 "Q",
#                 "N",
#                 "f",
#                 "d",
#             ]:
#                 if not (big == "!" and tpe in ("?", "b", "B")):
#                     source = """input x

# x {0} stack
# """.format(
#                         rep + big + tpe + "->"
#                     )
#                     vm32 = awkward.forth.ForthMachine32(source)
#                     assert vm32.decompiled == source
#                     source = """input x
# output y int32

# x {0} y
# """.format(
#                         rep + big + tpe + "->"
#                     )
#                     vm32 = awkward.forth.ForthMachine32(source)
#                     assert vm32.decompiled == source
#                     del vm32


def test_read_compilation():
    for i in range(100):
        vm32 = awkward.forth.ForthMachine32("1 2 3 4")

def test_read_compilation_2():
    test_read_compilation()


def test_read_compilation_3():
    test_read_compilation()


def test_read_compilation_4():
    test_read_compilation()


def test_read_compilation_5():
    test_read_compilation()


def test_read_compilation_6():
    test_read_compilation()


def test_read_compilation_7():
    test_read_compilation()


def test_read_compilation_8():
    test_read_compilation()


def test_read_compilation_9():
    test_read_compilation()


def test_read_compilation_10():
    test_read_compilation()


def test_read_compilation_11():
    test_read_compilation()


def test_read_compilation_12():
    test_read_compilation()


def test_read_compilation_13():
    test_read_compilation()


def test_read_compilation_14():
    test_read_compilation()


def test_read_compilation_15():
    test_read_compilation()


def test_read_compilation_16():
    test_read_compilation()


def test_read_compilation_17():
    test_read_compilation()


def test_read_compilation_18():
    test_read_compilation()


def test_read_compilation_19():
    test_read_compilation()


def test_read_compilation_20():
    test_read_compilation()


def test_read_compilation_21():
    test_read_compilation()


def test_read_compilation_22():
    test_read_compilation()


def test_read_compilation_23():
    test_read_compilation()


def test_read_compilation_24():
    test_read_compilation()


def test_read_compilation_25():
    test_read_compilation()


def test_read_compilation_26():
    test_read_compilation()


def test_read_compilation_27():
    test_read_compilation()


def test_read_compilation_28():
    test_read_compilation()


def test_read_compilation_29():
    test_read_compilation()


def test_read_compilation_30():
    test_read_compilation()


def test_read_compilation_31():
    test_read_compilation()


def test_read_compilation_32():
    test_read_compilation()


def test_read_compilation_33():
    test_read_compilation()


def test_read_compilation_34():
    test_read_compilation()


def test_read_compilation_35():
    test_read_compilation()


def test_read_compilation_36():
    test_read_compilation()


def test_read_compilation_37():
    test_read_compilation()


def test_read_compilation_38():
    test_read_compilation()


def test_read_compilation_39():
    test_read_compilation()


def test_read_compilation_40():
    test_read_compilation()


def test_read_compilation_41():
    test_read_compilation()


def test_read_compilation_42():
    test_read_compilation()


def test_read_compilation_43():
    test_read_compilation()


def test_read_compilation_44():
    test_read_compilation()


def test_read_compilation_45():
    test_read_compilation()


def test_read_compilation_46():
    test_read_compilation()


def test_read_compilation_47():
    test_read_compilation()


def test_read_compilation_48():
    test_read_compilation()


def test_read_compilation_49():
    test_read_compilation()


def test_read_compilation_50():
    test_read_compilation()


def test_read_compilation_51():
    test_read_compilation()


def test_read_compilation_52():
    test_read_compilation()


def test_read_compilation_53():
    test_read_compilation()


def test_read_compilation_54():
    test_read_compilation()


def test_read_compilation_55():
    test_read_compilation()


def test_read_compilation_56():
    test_read_compilation()


def test_read_compilation_57():
    test_read_compilation()


def test_read_compilation_58():
    test_read_compilation()


def test_read_compilation_59():
    test_read_compilation()


def test_read_compilation_60():
    test_read_compilation()


def test_read_compilation_61():
    test_read_compilation()


def test_read_compilation_62():
    test_read_compilation()


def test_read_compilation_63():
    test_read_compilation()


def test_read_compilation_64():
    test_read_compilation()


def test_read_compilation_65():
    test_read_compilation()


def test_read_compilation_66():
    test_read_compilation()


def test_read_compilation_67():
    test_read_compilation()


def test_read_compilation_68():
    test_read_compilation()


def test_read_compilation_69():
    test_read_compilation()


def test_read_compilation_70():
    test_read_compilation()


def test_read_compilation_71():
    test_read_compilation()


def test_read_compilation_72():
    test_read_compilation()


def test_read_compilation_73():
    test_read_compilation()


def test_read_compilation_74():
    test_read_compilation()


def test_read_compilation_75():
    test_read_compilation()


def test_read_compilation_76():
    test_read_compilation()


def test_read_compilation_77():
    test_read_compilation()


def test_read_compilation_78():
    test_read_compilation()


def test_read_compilation_79():
    test_read_compilation()


def test_read_compilation_80():
    test_read_compilation()


def test_read_compilation_81():
    test_read_compilation()


def test_read_compilation_82():
    test_read_compilation()


def test_read_compilation_83():
    test_read_compilation()


def test_read_compilation_84():
    test_read_compilation()


def test_read_compilation_85():
    test_read_compilation()


def test_read_compilation_86():
    test_read_compilation()


def test_read_compilation_87():
    test_read_compilation()


def test_read_compilation_88():
    test_read_compilation()


def test_read_compilation_89():
    test_read_compilation()


def test_read_compilation_90():
    test_read_compilation()


def test_read_compilation_91():
    test_read_compilation()


def test_read_compilation_92():
    test_read_compilation()


def test_read_compilation_93():
    test_read_compilation()


def test_read_compilation_94():
    test_read_compilation()


def test_read_compilation_95():
    test_read_compilation()


def test_read_compilation_96():
    test_read_compilation()


def test_read_compilation_97():
    test_read_compilation()


def test_read_compilation_98():
    test_read_compilation()


def test_read_compilation_99():
    test_read_compilation()


def test_read_compilation_100():
    test_read_compilation()


def test_read_compilation_101():
    test_read_compilation()


def test_read_compilation_102():
    test_read_compilation()


def test_read_compilation_103():
    test_read_compilation()


def test_read_compilation_104():
    test_read_compilation()


def test_read_compilation_105():
    test_read_compilation()


def test_read_compilation_106():
    test_read_compilation()


def test_read_compilation_107():
    test_read_compilation()


def test_read_compilation_108():
    test_read_compilation()


def test_read_compilation_109():
    test_read_compilation()


def test_read_compilation_110():
    test_read_compilation()


def test_read_compilation_111():
    test_read_compilation()


def test_read_compilation_112():
    test_read_compilation()


def test_read_compilation_113():
    test_read_compilation()


def test_read_compilation_114():
    test_read_compilation()


def test_read_compilation_115():
    test_read_compilation()


def test_read_compilation_116():
    test_read_compilation()


def test_read_compilation_117():
    test_read_compilation()


def test_read_compilation_118():
    test_read_compilation()


def test_read_compilation_119():
    test_read_compilation()


def test_read_compilation_120():
    test_read_compilation()


def test_read_compilation_121():
    test_read_compilation()


def test_read_compilation_122():
    test_read_compilation()


def test_read_compilation_123():
    test_read_compilation()


def test_read_compilation_124():
    test_read_compilation()


def test_read_compilation_125():
    test_read_compilation()


def test_read_compilation_126():
    test_read_compilation()


def test_read_compilation_127():
    test_read_compilation()


def test_read_compilation_128():
    test_read_compilation()


def test_read_compilation_129():
    test_read_compilation()


def test_read_compilation_130():
    test_read_compilation()


def test_read_compilation_131():
    test_read_compilation()


def test_read_compilation_132():
    test_read_compilation()


def test_read_compilation_133():
    test_read_compilation()


def test_read_compilation_134():
    test_read_compilation()


def test_read_compilation_135():
    test_read_compilation()


def test_read_compilation_136():
    test_read_compilation()


def test_read_compilation_137():
    test_read_compilation()


def test_read_compilation_138():
    test_read_compilation()


def test_read_compilation_139():
    test_read_compilation()


def test_read_compilation_140():
    test_read_compilation()


def test_read_compilation_141():
    test_read_compilation()


def test_read_compilation_142():
    test_read_compilation()


def test_read_compilation_143():
    test_read_compilation()


def test_read_compilation_144():
    test_read_compilation()


def test_read_compilation_145():
    test_read_compilation()


def test_read_compilation_146():
    test_read_compilation()


def test_read_compilation_147():
    test_read_compilation()


def test_read_compilation_148():
    test_read_compilation()


def test_read_compilation_149():
    test_read_compilation()


def test_read_compilation_150():
    test_read_compilation()


def test_read_compilation_151():
    test_read_compilation()


def test_read_compilation_152():
    test_read_compilation()


def test_read_compilation_153():
    test_read_compilation()


def test_read_compilation_154():
    test_read_compilation()


def test_read_compilation_155():
    test_read_compilation()


def test_read_compilation_156():
    test_read_compilation()


def test_read_compilation_157():
    test_read_compilation()


def test_read_compilation_158():
    test_read_compilation()


def test_read_compilation_159():
    test_read_compilation()


def test_read_compilation_160():
    test_read_compilation()


def test_read_compilation_161():
    test_read_compilation()


def test_read_compilation_162():
    test_read_compilation()


def test_read_compilation_163():
    test_read_compilation()


def test_read_compilation_164():
    test_read_compilation()


def test_read_compilation_165():
    test_read_compilation()


def test_read_compilation_166():
    test_read_compilation()


def test_read_compilation_167():
    test_read_compilation()


def test_read_compilation_168():
    test_read_compilation()


def test_read_compilation_169():
    test_read_compilation()


def test_read_compilation_170():
    test_read_compilation()


def test_read_compilation_171():
    test_read_compilation()


def test_read_compilation_172():
    test_read_compilation()


def test_read_compilation_173():
    test_read_compilation()


def test_read_compilation_174():
    test_read_compilation()


def test_read_compilation_175():
    test_read_compilation()


def test_read_compilation_176():
    test_read_compilation()


def test_read_compilation_177():
    test_read_compilation()


def test_read_compilation_178():
    test_read_compilation()


def test_read_compilation_179():
    test_read_compilation()


def test_read_compilation_180():
    test_read_compilation()


def test_read_compilation_181():
    test_read_compilation()


def test_read_compilation_182():
    test_read_compilation()


def test_read_compilation_183():
    test_read_compilation()


def test_read_compilation_184():
    test_read_compilation()


def test_read_compilation_185():
    test_read_compilation()


def test_read_compilation_186():
    test_read_compilation()


def test_read_compilation_187():
    test_read_compilation()


def test_read_compilation_188():
    test_read_compilation()


def test_read_compilation_189():
    test_read_compilation()


def test_read_compilation_190():
    test_read_compilation()


def test_read_compilation_191():
    test_read_compilation()


def test_read_compilation_192():
    test_read_compilation()


def test_read_compilation_193():
    test_read_compilation()


def test_read_compilation_194():
    test_read_compilation()


def test_read_compilation_195():
    test_read_compilation()


def test_read_compilation_196():
    test_read_compilation()


def test_read_compilation_197():
    test_read_compilation()


def test_read_compilation_198():
    test_read_compilation()


def test_read_compilation_199():
    test_read_compilation()


def test_read_compilation_200():
    test_read_compilation()


def test_read_compilation_201():
    test_read_compilation()


def test_read_compilation_202():
    test_read_compilation()


def test_read_compilation_203():
    test_read_compilation()


def test_read_compilation_204():
    test_read_compilation()


def test_read_compilation_205():
    test_read_compilation()


def test_read_compilation_206():
    test_read_compilation()


def test_read_compilation_207():
    test_read_compilation()


def test_read_compilation_208():
    test_read_compilation()


def test_read_compilation_209():
    test_read_compilation()


def test_read_compilation_210():
    test_read_compilation()


def test_read_compilation_211():
    test_read_compilation()


def test_read_compilation_212():
    test_read_compilation()


def test_read_compilation_213():
    test_read_compilation()


def test_read_compilation_214():
    test_read_compilation()


def test_read_compilation_215():
    test_read_compilation()


def test_read_compilation_216():
    test_read_compilation()


def test_read_compilation_217():
    test_read_compilation()


def test_read_compilation_218():
    test_read_compilation()


def test_read_compilation_219():
    test_read_compilation()


def test_read_compilation_220():
    test_read_compilation()


def test_read_compilation_221():
    test_read_compilation()


def test_read_compilation_222():
    test_read_compilation()


def test_read_compilation_223():
    test_read_compilation()


def test_read_compilation_224():
    test_read_compilation()


def test_read_compilation_225():
    test_read_compilation()


def test_read_compilation_226():
    test_read_compilation()


def test_read_compilation_227():
    test_read_compilation()


def test_read_compilation_228():
    test_read_compilation()


def test_read_compilation_229():
    test_read_compilation()


def test_read_compilation_230():
    test_read_compilation()


def test_read_compilation_231():
    test_read_compilation()


def test_read_compilation_232():
    test_read_compilation()


def test_read_compilation_233():
    test_read_compilation()


def test_read_compilation_234():
    test_read_compilation()


def test_read_compilation_235():
    test_read_compilation()


def test_read_compilation_236():
    test_read_compilation()


def test_read_compilation_237():
    test_read_compilation()


def test_read_compilation_238():
    test_read_compilation()


def test_read_compilation_239():
    test_read_compilation()


def test_read_compilation_240():
    test_read_compilation()


def test_read_compilation_241():
    test_read_compilation()


def test_read_compilation_242():
    test_read_compilation()


def test_read_compilation_243():
    test_read_compilation()


def test_read_compilation_244():
    test_read_compilation()


def test_read_compilation_245():
    test_read_compilation()


def test_read_compilation_246():
    test_read_compilation()


def test_read_compilation_247():
    test_read_compilation()


def test_read_compilation_248():
    test_read_compilation()


def test_read_compilation_249():
    test_read_compilation()


def test_read_compilation_250():
    test_read_compilation()


def test_read_compilation_251():
    test_read_compilation()


def test_read_compilation_252():
    test_read_compilation()


def test_read_compilation_253():
    test_read_compilation()


def test_read_compilation_254():
    test_read_compilation()


def test_read_compilation_255():
    test_read_compilation()


def test_read_compilation_256():
    test_read_compilation()


def test_read_compilation_257():
    test_read_compilation()


def test_read_compilation_258():
    test_read_compilation()


def test_read_compilation_259():
    test_read_compilation()


def test_read_compilation_260():
    test_read_compilation()


def test_read_compilation_261():
    test_read_compilation()


def test_read_compilation_262():
    test_read_compilation()


def test_read_compilation_263():
    test_read_compilation()


def test_read_compilation_264():
    test_read_compilation()


def test_read_compilation_265():
    test_read_compilation()


def test_read_compilation_266():
    test_read_compilation()


def test_read_compilation_267():
    test_read_compilation()


def test_read_compilation_268():
    test_read_compilation()


def test_read_compilation_269():
    test_read_compilation()


def test_read_compilation_270():
    test_read_compilation()


def test_read_compilation_271():
    test_read_compilation()


def test_read_compilation_272():
    test_read_compilation()


def test_read_compilation_273():
    test_read_compilation()


def test_read_compilation_274():
    test_read_compilation()


def test_read_compilation_275():
    test_read_compilation()


def test_read_compilation_276():
    test_read_compilation()


def test_read_compilation_277():
    test_read_compilation()


def test_read_compilation_278():
    test_read_compilation()


def test_read_compilation_279():
    test_read_compilation()


def test_read_compilation_280():
    test_read_compilation()


def test_read_compilation_281():
    test_read_compilation()


def test_read_compilation_282():
    test_read_compilation()


def test_read_compilation_283():
    test_read_compilation()


def test_read_compilation_284():
    test_read_compilation()


def test_read_compilation_285():
    test_read_compilation()


def test_read_compilation_286():
    test_read_compilation()


def test_read_compilation_287():
    test_read_compilation()


def test_read_compilation_288():
    test_read_compilation()


def test_read_compilation_289():
    test_read_compilation()


def test_read_compilation_290():
    test_read_compilation()


def test_read_compilation_291():
    test_read_compilation()


def test_read_compilation_292():
    test_read_compilation()


def test_read_compilation_293():
    test_read_compilation()


def test_read_compilation_294():
    test_read_compilation()


def test_read_compilation_295():
    test_read_compilation()


def test_read_compilation_296():
    test_read_compilation()


def test_read_compilation_297():
    test_read_compilation()


def test_read_compilation_298():
    test_read_compilation()


def test_read_compilation_299():
    test_read_compilation()


def test_read_compilation_300():
    test_read_compilation()


def test_read_compilation_301():
    test_read_compilation()


def test_read_compilation_302():
    test_read_compilation()


def test_read_compilation_303():
    test_read_compilation()


def test_read_compilation_304():
    test_read_compilation()


def test_read_compilation_305():
    test_read_compilation()


def test_read_compilation_306():
    test_read_compilation()


def test_read_compilation_307():
    test_read_compilation()


def test_read_compilation_308():
    test_read_compilation()


def test_read_compilation_309():
    test_read_compilation()


def test_read_compilation_310():
    test_read_compilation()


def test_read_compilation_311():
    test_read_compilation()


def test_read_compilation_312():
    test_read_compilation()


def test_read_compilation_313():
    test_read_compilation()


def test_read_compilation_314():
    test_read_compilation()


def test_read_compilation_315():
    test_read_compilation()


def test_read_compilation_316():
    test_read_compilation()


def test_read_compilation_317():
    test_read_compilation()


def test_read_compilation_318():
    test_read_compilation()


def test_read_compilation_319():
    test_read_compilation()


def test_read_compilation_320():
    test_read_compilation()


def test_read_compilation_321():
    test_read_compilation()


def test_read_compilation_322():
    test_read_compilation()


def test_read_compilation_323():
    test_read_compilation()


def test_read_compilation_324():
    test_read_compilation()


def test_read_compilation_325():
    test_read_compilation()


def test_read_compilation_326():
    test_read_compilation()


def test_read_compilation_327():
    test_read_compilation()


def test_read_compilation_328():
    test_read_compilation()


def test_read_compilation_329():
    test_read_compilation()


def test_read_compilation_330():
    test_read_compilation()


def test_read_compilation_331():
    test_read_compilation()


def test_read_compilation_332():
    test_read_compilation()


def test_read_compilation_333():
    test_read_compilation()


def test_read_compilation_334():
    test_read_compilation()


def test_read_compilation_335():
    test_read_compilation()


def test_read_compilation_336():
    test_read_compilation()


def test_read_compilation_337():
    test_read_compilation()


def test_read_compilation_338():
    test_read_compilation()


def test_read_compilation_339():
    test_read_compilation()


def test_read_compilation_340():
    test_read_compilation()


def test_read_compilation_341():
    test_read_compilation()


def test_read_compilation_342():
    test_read_compilation()


def test_read_compilation_343():
    test_read_compilation()


def test_read_compilation_344():
    test_read_compilation()


def test_read_compilation_345():
    test_read_compilation()


def test_read_compilation_346():
    test_read_compilation()


def test_read_compilation_347():
    test_read_compilation()


def test_read_compilation_348():
    test_read_compilation()


def test_read_compilation_349():
    test_read_compilation()


def test_read_compilation_350():
    test_read_compilation()


def test_read_compilation_351():
    test_read_compilation()


def test_read_compilation_352():
    test_read_compilation()


def test_read_compilation_353():
    test_read_compilation()


def test_read_compilation_354():
    test_read_compilation()


def test_read_compilation_355():
    test_read_compilation()


def test_read_compilation_356():
    test_read_compilation()


def test_read_compilation_357():
    test_read_compilation()


def test_read_compilation_358():
    test_read_compilation()


def test_read_compilation_359():
    test_read_compilation()


def test_read_compilation_360():
    test_read_compilation()


def test_read_compilation_361():
    test_read_compilation()


def test_read_compilation_362():
    test_read_compilation()


def test_read_compilation_363():
    test_read_compilation()


def test_read_compilation_364():
    test_read_compilation()


def test_read_compilation_365():
    test_read_compilation()


def test_read_compilation_366():
    test_read_compilation()


def test_read_compilation_367():
    test_read_compilation()


def test_read_compilation_368():
    test_read_compilation()


def test_read_compilation_369():
    test_read_compilation()


def test_read_compilation_370():
    test_read_compilation()


def test_read_compilation_371():
    test_read_compilation()


def test_read_compilation_372():
    test_read_compilation()


def test_read_compilation_373():
    test_read_compilation()


def test_read_compilation_374():
    test_read_compilation()


def test_read_compilation_375():
    test_read_compilation()


def test_read_compilation_376():
    test_read_compilation()


def test_read_compilation_377():
    test_read_compilation()


def test_read_compilation_378():
    test_read_compilation()


def test_read_compilation_379():
    test_read_compilation()


def test_read_compilation_380():
    test_read_compilation()


def test_read_compilation_381():
    test_read_compilation()


def test_read_compilation_382():
    test_read_compilation()


def test_read_compilation_383():
    test_read_compilation()


def test_read_compilation_384():
    test_read_compilation()


def test_read_compilation_385():
    test_read_compilation()


def test_read_compilation_386():
    test_read_compilation()


def test_read_compilation_387():
    test_read_compilation()


def test_read_compilation_388():
    test_read_compilation()


def test_read_compilation_389():
    test_read_compilation()


def test_read_compilation_390():
    test_read_compilation()


def test_read_compilation_391():
    test_read_compilation()


def test_read_compilation_392():
    test_read_compilation()


def test_read_compilation_393():
    test_read_compilation()


def test_read_compilation_394():
    test_read_compilation()


def test_read_compilation_395():
    test_read_compilation()


def test_read_compilation_396():
    test_read_compilation()


def test_read_compilation_397():
    test_read_compilation()


def test_read_compilation_398():
    test_read_compilation()


def test_read_compilation_399():
    test_read_compilation()


def test_read_compilation_400():
    test_read_compilation()


def test_read_compilation_401():
    test_read_compilation()


def test_read_compilation_402():
    test_read_compilation()


def test_read_compilation_403():
    test_read_compilation()


def test_read_compilation_404():
    test_read_compilation()


def test_read_compilation_405():
    test_read_compilation()


def test_read_compilation_406():
    test_read_compilation()


def test_read_compilation_407():
    test_read_compilation()


def test_read_compilation_408():
    test_read_compilation()


def test_read_compilation_409():
    test_read_compilation()


def test_read_compilation_410():
    test_read_compilation()


def test_read_compilation_411():
    test_read_compilation()


def test_read_compilation_412():
    test_read_compilation()


def test_read_compilation_413():
    test_read_compilation()


def test_read_compilation_414():
    test_read_compilation()


def test_read_compilation_415():
    test_read_compilation()


def test_read_compilation_416():
    test_read_compilation()


def test_read_compilation_417():
    test_read_compilation()


def test_read_compilation_418():
    test_read_compilation()


def test_read_compilation_419():
    test_read_compilation()


def test_read_compilation_420():
    test_read_compilation()


def test_read_compilation_421():
    test_read_compilation()


def test_read_compilation_422():
    test_read_compilation()


def test_read_compilation_423():
    test_read_compilation()


def test_read_compilation_424():
    test_read_compilation()


def test_read_compilation_425():
    test_read_compilation()


def test_read_compilation_426():
    test_read_compilation()


def test_read_compilation_427():
    test_read_compilation()


def test_read_compilation_428():
    test_read_compilation()


def test_read_compilation_429():
    test_read_compilation()


def test_read_compilation_430():
    test_read_compilation()


def test_read_compilation_431():
    test_read_compilation()


def test_read_compilation_432():
    test_read_compilation()


def test_read_compilation_433():
    test_read_compilation()


def test_read_compilation_434():
    test_read_compilation()


def test_read_compilation_435():
    test_read_compilation()


def test_read_compilation_436():
    test_read_compilation()


def test_read_compilation_437():
    test_read_compilation()


def test_read_compilation_438():
    test_read_compilation()


def test_read_compilation_439():
    test_read_compilation()


def test_read_compilation_440():
    test_read_compilation()


def test_read_compilation_441():
    test_read_compilation()


def test_read_compilation_442():
    test_read_compilation()


def test_read_compilation_443():
    test_read_compilation()


def test_read_compilation_444():
    test_read_compilation()


def test_read_compilation_445():
    test_read_compilation()


def test_read_compilation_446():
    test_read_compilation()


def test_read_compilation_447():
    test_read_compilation()


def test_read_compilation_448():
    test_read_compilation()


def test_read_compilation_449():
    test_read_compilation()


def test_read_compilation_450():
    test_read_compilation()


def test_read_compilation_451():
    test_read_compilation()


def test_read_compilation_452():
    test_read_compilation()


def test_read_compilation_453():
    test_read_compilation()


def test_read_compilation_454():
    test_read_compilation()


def test_read_compilation_455():
    test_read_compilation()


def test_read_compilation_456():
    test_read_compilation()


def test_read_compilation_457():
    test_read_compilation()


def test_read_compilation_458():
    test_read_compilation()


def test_read_compilation_459():
    test_read_compilation()


def test_read_compilation_460():
    test_read_compilation()


def test_read_compilation_461():
    test_read_compilation()


def test_read_compilation_462():
    test_read_compilation()


def test_read_compilation_463():
    test_read_compilation()


def test_read_compilation_464():
    test_read_compilation()


def test_read_compilation_465():
    test_read_compilation()


def test_read_compilation_466():
    test_read_compilation()


def test_read_compilation_467():
    test_read_compilation()


def test_read_compilation_468():
    test_read_compilation()


def test_read_compilation_469():
    test_read_compilation()


def test_read_compilation_470():
    test_read_compilation()


def test_read_compilation_471():
    test_read_compilation()


def test_read_compilation_472():
    test_read_compilation()


def test_read_compilation_473():
    test_read_compilation()


def test_read_compilation_474():
    test_read_compilation()


def test_read_compilation_475():
    test_read_compilation()


def test_read_compilation_476():
    test_read_compilation()


def test_read_compilation_477():
    test_read_compilation()


def test_read_compilation_478():
    test_read_compilation()


def test_read_compilation_479():
    test_read_compilation()


def test_read_compilation_480():
    test_read_compilation()


def test_read_compilation_481():
    test_read_compilation()


def test_read_compilation_482():
    test_read_compilation()


def test_read_compilation_483():
    test_read_compilation()


def test_read_compilation_484():
    test_read_compilation()


def test_read_compilation_485():
    test_read_compilation()


def test_read_compilation_486():
    test_read_compilation()


def test_read_compilation_487():
    test_read_compilation()


def test_read_compilation_488():
    test_read_compilation()


def test_read_compilation_489():
    test_read_compilation()


def test_read_compilation_490():
    test_read_compilation()


def test_read_compilation_491():
    test_read_compilation()


def test_read_compilation_492():
    test_read_compilation()


def test_read_compilation_493():
    test_read_compilation()


def test_read_compilation_494():
    test_read_compilation()


def test_read_compilation_495():
    test_read_compilation()


def test_read_compilation_496():
    test_read_compilation()


def test_read_compilation_497():
    test_read_compilation()


def test_read_compilation_498():
    test_read_compilation()


def test_read_compilation_499():
    test_read_compilation()


def test_read_compilation_500():
    test_read_compilation()


def test_read_compilation_501():
    test_read_compilation()


def test_read_compilation_502():
    test_read_compilation()


def test_read_compilation_503():
    test_read_compilation()


def test_read_compilation_504():
    test_read_compilation()


def test_read_compilation_505():
    test_read_compilation()


def test_read_compilation_506():
    test_read_compilation()


def test_read_compilation_507():
    test_read_compilation()


def test_read_compilation_508():
    test_read_compilation()


def test_read_compilation_509():
    test_read_compilation()


def test_read_compilation_510():
    test_read_compilation()


def test_read_compilation_511():
    test_read_compilation()


def test_read_compilation_512():
    test_read_compilation()


def test_read_compilation_513():
    test_read_compilation()


def test_read_compilation_514():
    test_read_compilation()


def test_read_compilation_515():
    test_read_compilation()


def test_read_compilation_516():
    test_read_compilation()


def test_read_compilation_517():
    test_read_compilation()


def test_read_compilation_518():
    test_read_compilation()


def test_read_compilation_519():
    test_read_compilation()


def test_read_compilation_520():
    test_read_compilation()


def test_read_compilation_521():
    test_read_compilation()


def test_read_compilation_522():
    test_read_compilation()


def test_read_compilation_523():
    test_read_compilation()


def test_read_compilation_524():
    test_read_compilation()


def test_read_compilation_525():
    test_read_compilation()


def test_read_compilation_526():
    test_read_compilation()


def test_read_compilation_527():
    test_read_compilation()


def test_read_compilation_528():
    test_read_compilation()


def test_read_compilation_529():
    test_read_compilation()


def test_read_compilation_530():
    test_read_compilation()


def test_read_compilation_531():
    test_read_compilation()


def test_read_compilation_532():
    test_read_compilation()


def test_read_compilation_533():
    test_read_compilation()


def test_read_compilation_534():
    test_read_compilation()


def test_read_compilation_535():
    test_read_compilation()


def test_read_compilation_536():
    test_read_compilation()


def test_read_compilation_537():
    test_read_compilation()


def test_read_compilation_538():
    test_read_compilation()


def test_read_compilation_539():
    test_read_compilation()


def test_read_compilation_540():
    test_read_compilation()


def test_read_compilation_541():
    test_read_compilation()


def test_read_compilation_542():
    test_read_compilation()


def test_read_compilation_543():
    test_read_compilation()


def test_read_compilation_544():
    test_read_compilation()


def test_read_compilation_545():
    test_read_compilation()


def test_read_compilation_546():
    test_read_compilation()


def test_read_compilation_547():
    test_read_compilation()


def test_read_compilation_548():
    test_read_compilation()


def test_read_compilation_549():
    test_read_compilation()


def test_read_compilation_550():
    test_read_compilation()


def test_read_compilation_551():
    test_read_compilation()


def test_read_compilation_552():
    test_read_compilation()


def test_read_compilation_553():
    test_read_compilation()


def test_read_compilation_554():
    test_read_compilation()


def test_read_compilation_555():
    test_read_compilation()


def test_read_compilation_556():
    test_read_compilation()


def test_read_compilation_557():
    test_read_compilation()


def test_read_compilation_558():
    test_read_compilation()


def test_read_compilation_559():
    test_read_compilation()


def test_read_compilation_560():
    test_read_compilation()


def test_read_compilation_561():
    test_read_compilation()


def test_read_compilation_562():
    test_read_compilation()


def test_read_compilation_563():
    test_read_compilation()


def test_read_compilation_564():
    test_read_compilation()


def test_read_compilation_565():
    test_read_compilation()


def test_read_compilation_566():
    test_read_compilation()


def test_read_compilation_567():
    test_read_compilation()


def test_read_compilation_568():
    test_read_compilation()


def test_read_compilation_569():
    test_read_compilation()


def test_read_compilation_570():
    test_read_compilation()


def test_read_compilation_571():
    test_read_compilation()


def test_read_compilation_572():
    test_read_compilation()


def test_read_compilation_573():
    test_read_compilation()


def test_read_compilation_574():
    test_read_compilation()


def test_read_compilation_575():
    test_read_compilation()


def test_read_compilation_576():
    test_read_compilation()


def test_read_compilation_577():
    test_read_compilation()


def test_read_compilation_578():
    test_read_compilation()


def test_read_compilation_579():
    test_read_compilation()


def test_read_compilation_580():
    test_read_compilation()


def test_read_compilation_581():
    test_read_compilation()


def test_read_compilation_582():
    test_read_compilation()


def test_read_compilation_583():
    test_read_compilation()


def test_read_compilation_584():
    test_read_compilation()


def test_read_compilation_585():
    test_read_compilation()


def test_read_compilation_586():
    test_read_compilation()


def test_read_compilation_587():
    test_read_compilation()


def test_read_compilation_588():
    test_read_compilation()


def test_read_compilation_589():
    test_read_compilation()


def test_read_compilation_590():
    test_read_compilation()


def test_read_compilation_591():
    test_read_compilation()


def test_read_compilation_592():
    test_read_compilation()


def test_read_compilation_593():
    test_read_compilation()


def test_read_compilation_594():
    test_read_compilation()


def test_read_compilation_595():
    test_read_compilation()


def test_read_compilation_596():
    test_read_compilation()


def test_read_compilation_597():
    test_read_compilation()


def test_read_compilation_598():
    test_read_compilation()


def test_read_compilation_599():
    test_read_compilation()


def test_read_compilation_600():
    test_read_compilation()


def test_read_compilation_601():
    test_read_compilation()


def test_read_compilation_602():
    test_read_compilation()


def test_read_compilation_603():
    test_read_compilation()


def test_read_compilation_604():
    test_read_compilation()


def test_read_compilation_605():
    test_read_compilation()


def test_read_compilation_606():
    test_read_compilation()


def test_read_compilation_607():
    test_read_compilation()


def test_read_compilation_608():
    test_read_compilation()


def test_read_compilation_609():
    test_read_compilation()


def test_read_compilation_610():
    test_read_compilation()


def test_read_compilation_611():
    test_read_compilation()


def test_read_compilation_612():
    test_read_compilation()


def test_read_compilation_613():
    test_read_compilation()


def test_read_compilation_614():
    test_read_compilation()


def test_read_compilation_615():
    test_read_compilation()


def test_read_compilation_616():
    test_read_compilation()


def test_read_compilation_617():
    test_read_compilation()


def test_read_compilation_618():
    test_read_compilation()


def test_read_compilation_619():
    test_read_compilation()


def test_read_compilation_620():
    test_read_compilation()


def test_read_compilation_621():
    test_read_compilation()


def test_read_compilation_622():
    test_read_compilation()


def test_read_compilation_623():
    test_read_compilation()


def test_read_compilation_624():
    test_read_compilation()


def test_read_compilation_625():
    test_read_compilation()


def test_read_compilation_626():
    test_read_compilation()


def test_read_compilation_627():
    test_read_compilation()


def test_read_compilation_628():
    test_read_compilation()


def test_read_compilation_629():
    test_read_compilation()


def test_read_compilation_630():
    test_read_compilation()


def test_read_compilation_631():
    test_read_compilation()


def test_read_compilation_632():
    test_read_compilation()


def test_read_compilation_633():
    test_read_compilation()


def test_read_compilation_634():
    test_read_compilation()


def test_read_compilation_635():
    test_read_compilation()


def test_read_compilation_636():
    test_read_compilation()


def test_read_compilation_637():
    test_read_compilation()


def test_read_compilation_638():
    test_read_compilation()


def test_read_compilation_639():
    test_read_compilation()


def test_read_compilation_640():
    test_read_compilation()


def test_read_compilation_641():
    test_read_compilation()


def test_read_compilation_642():
    test_read_compilation()


def test_read_compilation_643():
    test_read_compilation()


def test_read_compilation_644():
    test_read_compilation()


def test_read_compilation_645():
    test_read_compilation()


def test_read_compilation_646():
    test_read_compilation()


def test_read_compilation_647():
    test_read_compilation()


def test_read_compilation_648():
    test_read_compilation()


def test_read_compilation_649():
    test_read_compilation()


def test_read_compilation_650():
    test_read_compilation()


def test_read_compilation_651():
    test_read_compilation()


def test_read_compilation_652():
    test_read_compilation()


def test_read_compilation_653():
    test_read_compilation()


def test_read_compilation_654():
    test_read_compilation()


def test_read_compilation_655():
    test_read_compilation()


def test_read_compilation_656():
    test_read_compilation()


def test_read_compilation_657():
    test_read_compilation()


def test_read_compilation_658():
    test_read_compilation()


def test_read_compilation_659():
    test_read_compilation()


def test_read_compilation_660():
    test_read_compilation()


def test_read_compilation_661():
    test_read_compilation()


def test_read_compilation_662():
    test_read_compilation()


def test_read_compilation_663():
    test_read_compilation()


def test_read_compilation_664():
    test_read_compilation()


def test_read_compilation_665():
    test_read_compilation()


def test_read_compilation_666():
    test_read_compilation()


def test_read_compilation_667():
    test_read_compilation()


def test_read_compilation_668():
    test_read_compilation()


def test_read_compilation_669():
    test_read_compilation()


def test_read_compilation_670():
    test_read_compilation()


def test_read_compilation_671():
    test_read_compilation()


def test_read_compilation_672():
    test_read_compilation()


def test_read_compilation_673():
    test_read_compilation()


def test_read_compilation_674():
    test_read_compilation()


def test_read_compilation_675():
    test_read_compilation()


def test_read_compilation_676():
    test_read_compilation()


def test_read_compilation_677():
    test_read_compilation()


def test_read_compilation_678():
    test_read_compilation()


def test_read_compilation_679():
    test_read_compilation()


def test_read_compilation_680():
    test_read_compilation()


def test_read_compilation_681():
    test_read_compilation()


def test_read_compilation_682():
    test_read_compilation()


def test_read_compilation_683():
    test_read_compilation()


def test_read_compilation_684():
    test_read_compilation()


def test_read_compilation_685():
    test_read_compilation()


def test_read_compilation_686():
    test_read_compilation()


def test_read_compilation_687():
    test_read_compilation()


def test_read_compilation_688():
    test_read_compilation()


def test_read_compilation_689():
    test_read_compilation()


def test_read_compilation_690():
    test_read_compilation()


def test_read_compilation_691():
    test_read_compilation()


def test_read_compilation_692():
    test_read_compilation()


def test_read_compilation_693():
    test_read_compilation()


def test_read_compilation_694():
    test_read_compilation()


def test_read_compilation_695():
    test_read_compilation()


def test_read_compilation_696():
    test_read_compilation()


def test_read_compilation_697():
    test_read_compilation()


def test_read_compilation_698():
    test_read_compilation()


def test_read_compilation_699():
    test_read_compilation()


def test_read_compilation_700():
    test_read_compilation()


def test_read_compilation_701():
    test_read_compilation()


def test_read_compilation_702():
    test_read_compilation()


def test_read_compilation_703():
    test_read_compilation()


def test_read_compilation_704():
    test_read_compilation()


def test_read_compilation_705():
    test_read_compilation()


def test_read_compilation_706():
    test_read_compilation()


def test_read_compilation_707():
    test_read_compilation()


def test_read_compilation_708():
    test_read_compilation()


def test_read_compilation_709():
    test_read_compilation()


def test_read_compilation_710():
    test_read_compilation()


def test_read_compilation_711():
    test_read_compilation()


def test_read_compilation_712():
    test_read_compilation()


def test_read_compilation_713():
    test_read_compilation()


def test_read_compilation_714():
    test_read_compilation()


def test_read_compilation_715():
    test_read_compilation()


def test_read_compilation_716():
    test_read_compilation()


def test_read_compilation_717():
    test_read_compilation()


def test_read_compilation_718():
    test_read_compilation()


def test_read_compilation_719():
    test_read_compilation()


def test_read_compilation_720():
    test_read_compilation()


def test_read_compilation_721():
    test_read_compilation()


def test_read_compilation_722():
    test_read_compilation()


def test_read_compilation_723():
    test_read_compilation()


def test_read_compilation_724():
    test_read_compilation()


def test_read_compilation_725():
    test_read_compilation()


def test_read_compilation_726():
    test_read_compilation()


def test_read_compilation_727():
    test_read_compilation()


def test_read_compilation_728():
    test_read_compilation()


def test_read_compilation_729():
    test_read_compilation()


def test_read_compilation_730():
    test_read_compilation()


def test_read_compilation_731():
    test_read_compilation()


def test_read_compilation_732():
    test_read_compilation()


def test_read_compilation_733():
    test_read_compilation()


def test_read_compilation_734():
    test_read_compilation()


def test_read_compilation_735():
    test_read_compilation()


def test_read_compilation_736():
    test_read_compilation()


def test_read_compilation_737():
    test_read_compilation()


def test_read_compilation_738():
    test_read_compilation()


def test_read_compilation_739():
    test_read_compilation()


def test_read_compilation_740():
    test_read_compilation()


def test_read_compilation_741():
    test_read_compilation()


def test_read_compilation_742():
    test_read_compilation()


def test_read_compilation_743():
    test_read_compilation()


def test_read_compilation_744():
    test_read_compilation()


def test_read_compilation_745():
    test_read_compilation()


def test_read_compilation_746():
    test_read_compilation()


def test_read_compilation_747():
    test_read_compilation()


def test_read_compilation_748():
    test_read_compilation()


def test_read_compilation_749():
    test_read_compilation()


def test_read_compilation_750():
    test_read_compilation()


def test_read_compilation_751():
    test_read_compilation()


def test_read_compilation_752():
    test_read_compilation()


def test_read_compilation_753():
    test_read_compilation()


def test_read_compilation_754():
    test_read_compilation()


def test_read_compilation_755():
    test_read_compilation()


def test_read_compilation_756():
    test_read_compilation()


def test_read_compilation_757():
    test_read_compilation()


def test_read_compilation_758():
    test_read_compilation()


def test_read_compilation_759():
    test_read_compilation()


def test_read_compilation_760():
    test_read_compilation()


def test_read_compilation_761():
    test_read_compilation()


def test_read_compilation_762():
    test_read_compilation()


def test_read_compilation_763():
    test_read_compilation()


def test_read_compilation_764():
    test_read_compilation()


def test_read_compilation_765():
    test_read_compilation()


def test_read_compilation_766():
    test_read_compilation()


def test_read_compilation_767():
    test_read_compilation()


def test_read_compilation_768():
    test_read_compilation()


def test_read_compilation_769():
    test_read_compilation()


def test_read_compilation_770():
    test_read_compilation()


def test_read_compilation_771():
    test_read_compilation()


def test_read_compilation_772():
    test_read_compilation()


def test_read_compilation_773():
    test_read_compilation()


def test_read_compilation_774():
    test_read_compilation()


def test_read_compilation_775():
    test_read_compilation()


def test_read_compilation_776():
    test_read_compilation()


def test_read_compilation_777():
    test_read_compilation()


def test_read_compilation_778():
    test_read_compilation()


def test_read_compilation_779():
    test_read_compilation()


def test_read_compilation_780():
    test_read_compilation()


def test_read_compilation_781():
    test_read_compilation()


def test_read_compilation_782():
    test_read_compilation()


def test_read_compilation_783():
    test_read_compilation()


def test_read_compilation_784():
    test_read_compilation()


def test_read_compilation_785():
    test_read_compilation()


def test_read_compilation_786():
    test_read_compilation()


def test_read_compilation_787():
    test_read_compilation()


def test_read_compilation_788():
    test_read_compilation()


def test_read_compilation_789():
    test_read_compilation()


def test_read_compilation_790():
    test_read_compilation()


def test_read_compilation_791():
    test_read_compilation()


def test_read_compilation_792():
    test_read_compilation()


def test_read_compilation_793():
    test_read_compilation()


def test_read_compilation_794():
    test_read_compilation()


def test_read_compilation_795():
    test_read_compilation()


def test_read_compilation_796():
    test_read_compilation()


def test_read_compilation_797():
    test_read_compilation()


def test_read_compilation_798():
    test_read_compilation()


def test_read_compilation_799():
    test_read_compilation()


def test_read_compilation_800():
    test_read_compilation()


def test_read_compilation_801():
    test_read_compilation()


def test_read_compilation_802():
    test_read_compilation()


def test_read_compilation_803():
    test_read_compilation()


def test_read_compilation_804():
    test_read_compilation()


def test_read_compilation_805():
    test_read_compilation()


def test_read_compilation_806():
    test_read_compilation()


def test_read_compilation_807():
    test_read_compilation()


def test_read_compilation_808():
    test_read_compilation()


def test_read_compilation_809():
    test_read_compilation()


def test_read_compilation_810():
    test_read_compilation()


def test_read_compilation_811():
    test_read_compilation()


def test_read_compilation_812():
    test_read_compilation()


def test_read_compilation_813():
    test_read_compilation()


def test_read_compilation_814():
    test_read_compilation()


def test_read_compilation_815():
    test_read_compilation()


def test_read_compilation_816():
    test_read_compilation()


def test_read_compilation_817():
    test_read_compilation()


def test_read_compilation_818():
    test_read_compilation()


def test_read_compilation_819():
    test_read_compilation()


def test_read_compilation_820():
    test_read_compilation()


def test_read_compilation_821():
    test_read_compilation()


def test_read_compilation_822():
    test_read_compilation()


def test_read_compilation_823():
    test_read_compilation()


def test_read_compilation_824():
    test_read_compilation()


def test_read_compilation_825():
    test_read_compilation()


def test_read_compilation_826():
    test_read_compilation()


def test_read_compilation_827():
    test_read_compilation()


def test_read_compilation_828():
    test_read_compilation()


def test_read_compilation_829():
    test_read_compilation()


def test_read_compilation_830():
    test_read_compilation()


def test_read_compilation_831():
    test_read_compilation()


def test_read_compilation_832():
    test_read_compilation()


def test_read_compilation_833():
    test_read_compilation()


def test_read_compilation_834():
    test_read_compilation()


def test_read_compilation_835():
    test_read_compilation()


def test_read_compilation_836():
    test_read_compilation()


def test_read_compilation_837():
    test_read_compilation()


def test_read_compilation_838():
    test_read_compilation()


def test_read_compilation_839():
    test_read_compilation()


def test_read_compilation_840():
    test_read_compilation()


def test_read_compilation_841():
    test_read_compilation()


def test_read_compilation_842():
    test_read_compilation()


def test_read_compilation_843():
    test_read_compilation()


def test_read_compilation_844():
    test_read_compilation()


def test_read_compilation_845():
    test_read_compilation()


def test_read_compilation_846():
    test_read_compilation()


def test_read_compilation_847():
    test_read_compilation()


def test_read_compilation_848():
    test_read_compilation()


def test_read_compilation_849():
    test_read_compilation()


def test_read_compilation_850():
    test_read_compilation()


def test_read_compilation_851():
    test_read_compilation()


def test_read_compilation_852():
    test_read_compilation()


def test_read_compilation_853():
    test_read_compilation()


def test_read_compilation_854():
    test_read_compilation()


def test_read_compilation_855():
    test_read_compilation()


def test_read_compilation_856():
    test_read_compilation()


def test_read_compilation_857():
    test_read_compilation()


def test_read_compilation_858():
    test_read_compilation()


def test_read_compilation_859():
    test_read_compilation()


def test_read_compilation_860():
    test_read_compilation()


def test_read_compilation_861():
    test_read_compilation()


def test_read_compilation_862():
    test_read_compilation()


def test_read_compilation_863():
    test_read_compilation()


def test_read_compilation_864():
    test_read_compilation()


def test_read_compilation_865():
    test_read_compilation()


def test_read_compilation_866():
    test_read_compilation()


def test_read_compilation_867():
    test_read_compilation()


def test_read_compilation_868():
    test_read_compilation()


def test_read_compilation_869():
    test_read_compilation()


def test_read_compilation_870():
    test_read_compilation()


def test_read_compilation_871():
    test_read_compilation()


def test_read_compilation_872():
    test_read_compilation()


def test_read_compilation_873():
    test_read_compilation()


def test_read_compilation_874():
    test_read_compilation()


def test_read_compilation_875():
    test_read_compilation()


def test_read_compilation_876():
    test_read_compilation()


def test_read_compilation_877():
    test_read_compilation()


def test_read_compilation_878():
    test_read_compilation()


def test_read_compilation_879():
    test_read_compilation()


def test_read_compilation_880():
    test_read_compilation()


def test_read_compilation_881():
    test_read_compilation()


def test_read_compilation_882():
    test_read_compilation()


def test_read_compilation_883():
    test_read_compilation()


def test_read_compilation_884():
    test_read_compilation()


def test_read_compilation_885():
    test_read_compilation()


def test_read_compilation_886():
    test_read_compilation()


def test_read_compilation_887():
    test_read_compilation()


def test_read_compilation_888():
    test_read_compilation()


def test_read_compilation_889():
    test_read_compilation()


def test_read_compilation_890():
    test_read_compilation()


def test_read_compilation_891():
    test_read_compilation()


def test_read_compilation_892():
    test_read_compilation()


def test_read_compilation_893():
    test_read_compilation()


def test_read_compilation_894():
    test_read_compilation()


def test_read_compilation_895():
    test_read_compilation()


def test_read_compilation_896():
    test_read_compilation()


def test_read_compilation_897():
    test_read_compilation()


def test_read_compilation_898():
    test_read_compilation()


def test_read_compilation_899():
    test_read_compilation()


def test_read_compilation_900():
    test_read_compilation()


def test_read_compilation_901():
    test_read_compilation()


def test_read_compilation_902():
    test_read_compilation()


def test_read_compilation_903():
    test_read_compilation()


def test_read_compilation_904():
    test_read_compilation()


def test_read_compilation_905():
    test_read_compilation()


def test_read_compilation_906():
    test_read_compilation()


def test_read_compilation_907():
    test_read_compilation()


def test_read_compilation_908():
    test_read_compilation()


def test_read_compilation_909():
    test_read_compilation()


def test_read_compilation_910():
    test_read_compilation()


def test_read_compilation_911():
    test_read_compilation()


def test_read_compilation_912():
    test_read_compilation()


def test_read_compilation_913():
    test_read_compilation()


def test_read_compilation_914():
    test_read_compilation()


def test_read_compilation_915():
    test_read_compilation()


def test_read_compilation_916():
    test_read_compilation()


def test_read_compilation_917():
    test_read_compilation()


def test_read_compilation_918():
    test_read_compilation()


def test_read_compilation_919():
    test_read_compilation()


def test_read_compilation_920():
    test_read_compilation()


def test_read_compilation_921():
    test_read_compilation()


def test_read_compilation_922():
    test_read_compilation()


def test_read_compilation_923():
    test_read_compilation()


def test_read_compilation_924():
    test_read_compilation()


def test_read_compilation_925():
    test_read_compilation()


def test_read_compilation_926():
    test_read_compilation()


def test_read_compilation_927():
    test_read_compilation()


def test_read_compilation_928():
    test_read_compilation()


def test_read_compilation_929():
    test_read_compilation()


def test_read_compilation_930():
    test_read_compilation()


def test_read_compilation_931():
    test_read_compilation()


def test_read_compilation_932():
    test_read_compilation()


def test_read_compilation_933():
    test_read_compilation()


def test_read_compilation_934():
    test_read_compilation()


def test_read_compilation_935():
    test_read_compilation()


def test_read_compilation_936():
    test_read_compilation()


def test_read_compilation_937():
    test_read_compilation()


def test_read_compilation_938():
    test_read_compilation()


def test_read_compilation_939():
    test_read_compilation()


def test_read_compilation_940():
    test_read_compilation()


def test_read_compilation_941():
    test_read_compilation()


def test_read_compilation_942():
    test_read_compilation()


def test_read_compilation_943():
    test_read_compilation()


def test_read_compilation_944():
    test_read_compilation()


def test_read_compilation_945():
    test_read_compilation()


def test_read_compilation_946():
    test_read_compilation()


def test_read_compilation_947():
    test_read_compilation()


def test_read_compilation_948():
    test_read_compilation()


def test_read_compilation_949():
    test_read_compilation()


def test_read_compilation_950():
    test_read_compilation()


def test_read_compilation_951():
    test_read_compilation()


def test_read_compilation_952():
    test_read_compilation()


def test_read_compilation_953():
    test_read_compilation()


def test_read_compilation_954():
    test_read_compilation()


def test_read_compilation_955():
    test_read_compilation()


def test_read_compilation_956():
    test_read_compilation()


def test_read_compilation_957():
    test_read_compilation()


def test_read_compilation_958():
    test_read_compilation()


def test_read_compilation_959():
    test_read_compilation()


def test_read_compilation_960():
    test_read_compilation()


def test_read_compilation_961():
    test_read_compilation()


def test_read_compilation_962():
    test_read_compilation()


def test_read_compilation_963():
    test_read_compilation()


def test_read_compilation_964():
    test_read_compilation()


def test_read_compilation_965():
    test_read_compilation()


def test_read_compilation_966():
    test_read_compilation()


def test_read_compilation_967():
    test_read_compilation()


def test_read_compilation_968():
    test_read_compilation()


def test_read_compilation_969():
    test_read_compilation()


def test_read_compilation_970():
    test_read_compilation()


def test_read_compilation_971():
    test_read_compilation()


def test_read_compilation_972():
    test_read_compilation()


def test_read_compilation_973():
    test_read_compilation()


def test_read_compilation_974():
    test_read_compilation()


def test_read_compilation_975():
    test_read_compilation()


def test_read_compilation_976():
    test_read_compilation()


def test_read_compilation_977():
    test_read_compilation()


def test_read_compilation_978():
    test_read_compilation()


def test_read_compilation_979():
    test_read_compilation()


def test_read_compilation_980():
    test_read_compilation()


def test_read_compilation_981():
    test_read_compilation()


def test_read_compilation_982():
    test_read_compilation()


def test_read_compilation_983():
    test_read_compilation()


def test_read_compilation_984():
    test_read_compilation()


def test_read_compilation_985():
    test_read_compilation()


def test_read_compilation_986():
    test_read_compilation()


def test_read_compilation_987():
    test_read_compilation()


def test_read_compilation_988():
    test_read_compilation()


def test_read_compilation_989():
    test_read_compilation()


def test_read_compilation_990():
    test_read_compilation()


def test_read_compilation_991():
    test_read_compilation()


def test_read_compilation_992():
    test_read_compilation()


def test_read_compilation_993():
    test_read_compilation()


def test_read_compilation_994():
    test_read_compilation()


def test_read_compilation_995():
    test_read_compilation()


def test_read_compilation_996():
    test_read_compilation()


def test_read_compilation_997():
    test_read_compilation()


def test_read_compilation_998():
    test_read_compilation()


def test_read_compilation_999():
    test_read_compilation()


# def test_everything_else_compilation():
#     source = (
#         "\n".join(
#             [
#                 "dup",
#                 "drop",
#                 "swap",
#                 "over",
#                 "rot",
#                 "nip",
#                 "tuck",
#                 "+",
#                 "-",
#                 "*",
#                 "/",
#                 "mod",
#                 "/mod",
#                 "negate",
#                 "1+",
#                 "1-",
#                 "abs",
#                 "min",
#                 "max",
#                 "=",
#                 "<>",
#                 ">",
#                 ">=",
#                 "<",
#                 "<=",
#                 "0=",
#                 "invert",
#                 "and",
#                 "or",
#                 "xor",
#                 "lshift",
#                 "rshift",
#                 "false",
#                 "true",
#             ]
#         )
#         + "\n"
#     )
#     vm32 = awkward.forth.ForthMachine32(source)
#     assert vm32.decompiled == source


# def test_input_output():
#     vm32 = awkward.forth.ForthMachine32("input x output y int32")
#     vm32.begin({"x": np.array([1, 2, 3])})
#     assert isinstance(vm32["y"], ak.layout.NumpyArray)


# def test_stepping():
#     vm32 = awkward.forth.ForthMachine32("1 2 3 4")
#     vm32.begin()
#     assert vm32.stack == []
#     vm32.step()
#     assert vm32.stack == [1]
#     vm32.step()
#     assert vm32.stack == [1, 2]
#     vm32.step()
#     assert vm32.stack == [1, 2, 3]
#     vm32.step()
#     assert vm32.stack == [1, 2, 3, 4]
#     with pytest.raises(ValueError):
#         vm32.step()

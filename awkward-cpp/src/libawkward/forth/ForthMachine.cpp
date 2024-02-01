// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthMachine.cpp", line)

#include <sstream>
#include <stdexcept>
#include <chrono>

#include "awkward/forth/ForthMachine.h"

namespace awkward {
  // ABI version must be increased whenever any interpretation of the bytecode changes.
  #define ABI_VERSION 1

  // Instruction values are preprocessor macros to be equally usable in 32-bit and
  // 64-bit instruction sets.

  // parser flags (parsers are combined bitwise and then bit-inverted to be negative)
  #define READ_DIRECT 1
  #define READ_REPEATED 2
  #define READ_BIGENDIAN 4
  // parser sequential values (starting in the fourth bit)
  #define READ_MASK (~(-0x100) & (-0x8))
  #define READ_BOOL (0x8 * 1)
  #define READ_INT8 (0x8 * 2)
  #define READ_INT16 (0x8 * 3)
  #define READ_INT32 (0x8 * 4)
  #define READ_INT64 (0x8 * 5)
  #define READ_INTP (0x8 * 6)
  #define READ_UINT8 (0x8 * 7)
  #define READ_UINT16 (0x8 * 8)
  #define READ_UINT32 (0x8 * 9)
  #define READ_UINT64 (0x8 * 10)
  #define READ_UINTP (0x8 * 11)
  #define READ_FLOAT32 (0x8 * 12)
  #define READ_FLOAT64 (0x8 * 13)
  #define READ_VARINT (0x8 * 14)
  #define READ_ZIGZAG (0x8 * 15)
  #define READ_NBIT (0x8 * 16)
  #define READ_TEXTINT (0x8 * 17)
  #define READ_TEXTFLOAT (0x8 * 18)
  #define READ_QUOTEDSTR (0x8 * 19)

  // instructions from special parsing rules
  #define CODE_LITERAL 0
  #define CODE_HALT 1
  #define CODE_PAUSE 2
  #define CODE_IF 3
  #define CODE_IF_ELSE 4
  #define CODE_CASE_REGULAR 5
  #define CODE_DO 6
  #define CODE_DO_STEP 7
  #define CODE_AGAIN 8
  #define CODE_UNTIL 9
  #define CODE_WHILE 10
  #define CODE_EXIT 11
  #define CODE_PUT 12
  #define CODE_INC 13
  #define CODE_GET 14
  #define CODE_ENUM 15
  #define CODE_ENUMONLY 16
  #define CODE_PEEK 17
  #define CODE_LEN_INPUT 18
  #define CODE_POS 19
  #define CODE_END 20
  #define CODE_SEEK 21
  #define CODE_SKIP 22
  #define CODE_SKIPWS 23
  #define CODE_WRITE 24
  #define CODE_WRITE_ADD 25
  #define CODE_WRITE_DUP 26
  #define CODE_LEN_OUTPUT 27
  #define CODE_REWIND 28
  // generic builtin instructions
  #define CODE_STRING 29
  #define CODE_PRINT_STRING 30
  #define CODE_PRINT 31
  #define CODE_PRINT_CR 32
  #define CODE_PRINT_STACK 33
  #define CODE_I 34
  #define CODE_J 35
  #define CODE_K 36
  #define CODE_DUP 37
  #define CODE_DROP 38
  #define CODE_SWAP 39
  #define CODE_OVER 40
  #define CODE_ROT 41
  #define CODE_NIP 42
  #define CODE_TUCK 43
  #define CODE_ADD 44
  #define CODE_SUB 45
  #define CODE_MUL 46
  #define CODE_DIV 47
  #define CODE_MOD 48
  #define CODE_DIVMOD 49
  #define CODE_NEGATE 50
  #define CODE_ADD1 51
  #define CODE_SUB1 52
  #define CODE_ABS 53
  #define CODE_MIN 54
  #define CODE_MAX 55
  #define CODE_EQ 56
  #define CODE_NE 57
  #define CODE_GT 58
  #define CODE_GE 59
  #define CODE_LT 60
  #define CODE_LE 61
  #define CODE_EQ0 62
  #define CODE_INVERT 63
  #define CODE_AND 64
  #define CODE_OR 65
  #define CODE_XOR 66
  #define CODE_LSHIFT 67
  #define CODE_RSHIFT 68
  #define CODE_FALSE 69
  #define CODE_TRUE 70
  // beginning of the user-defined dictionary
  #define BOUND_DICTIONARY 71

  const std::set<std::string> reserved_words_({
    // comments
    "(", ")", "\\", "\n", "",
    // defining functions
    ":", ";", "recurse",
    // declaring globals
    "variable", "input", "output",
    // manipulate control flow externally
    "halt", "pause",
    // conditionals
    "if", "then", "else", "case", "of", "endof", "endcase",
    // loops
    "do", "loop", "+loop",
    "begin", "again", "until", "while", "repeat",
    // nonlocal exits
    "exit",
    // variable access
    "!", "+!", "@",
    // input actions
    "enum", "enumonly", "peek", "len", "pos", "end", "seek", "skip", "skipws",
    // output actions
    "<-", "+<-", "stack", "rewind",
    // print (for debugging)
    ".\"",
    // user defined strings
    "s\""
  });

  const std::set<std::string> input_parser_words_({
    // single little-endian
    "?->", "b->", "h->", "i->", "q->", "n->", "B->", "H->", "I->", "Q->", "N->",
    "f->", "d->", "varint->", "zigzag->", "textint->", "textfloat->", "quotedstr->",
    // single big-endian
    "!h->", "!i->", "!q->", "!n->", "!H->", "!I->", "!Q->", "!N->",
    "!f->", "!d->",
    // multiple little-endian
    "#?->", "#b->", "#h->", "#i->", "#q->", "#n->", "#B->", "#H->", "#I->", "#Q->", "#N->",
    "#f->", "#d->", "#varint->", "#zigzag->", "#textint->", "#textfloat->", "#quotedstr->",
    // multiple big-endian
    "#!h->", "#!i->", "#!q->", "#!n->", "#!H->", "#!I->", "#!Q->", "#!N->",
    "#!f->", "#!d->"
  });

  const std::map<std::string, util::dtype> output_dtype_words_({
    {"bool", util::dtype::boolean},
    {"int8", util::dtype::int8},
    {"int16", util::dtype::int16},
    {"int32", util::dtype::int32},
    {"int64", util::dtype::int64},
    {"uint8", util::dtype::uint8},
    {"uint16", util::dtype::uint16},
    {"uint32", util::dtype::uint32},
    {"uint64", util::dtype::uint64},
    {"float32", util::dtype::float32},
    {"float64", util::dtype::float64}
  });

  const std::map<std::string, int64_t> generic_builtin_words_({
    // print (for debugging)
    {".", CODE_PRINT},
    {"cr", CODE_PRINT_CR},
    {".s", CODE_PRINT_STACK},
    // loop variables
    {"i", CODE_I},
    {"j", CODE_J},
    {"k", CODE_K},
    // stack operations
    {"dup", CODE_DUP},
    {"drop", CODE_DROP},
    {"swap", CODE_SWAP},
    {"over", CODE_OVER},
    {"rot", CODE_ROT},
    {"nip", CODE_NIP},
    {"tuck", CODE_TUCK},
    // basic mathematical functions
    {"+", CODE_ADD},
    {"-", CODE_SUB},
    {"*", CODE_MUL},
    {"/", CODE_DIV},
    {"mod", CODE_MOD},
    {"/mod", CODE_DIVMOD},
    {"negate", CODE_NEGATE},
    {"1+", CODE_ADD1},
    {"1-", CODE_SUB1},
    {"abs", CODE_ABS},
    {"min", CODE_MIN},
    {"max", CODE_MAX},
    // comparisons
    {"=", CODE_EQ},
    {"<>", CODE_NE},
    {">", CODE_GT},
    {">=", CODE_GE},
    {"<", CODE_LT},
    {"<=", CODE_LE},
    {"0=", CODE_EQ0},
    // bitwise operations
    {"invert", CODE_INVERT},
    {"and", CODE_AND},
    {"or", CODE_OR},
    {"xor", CODE_XOR},
    {"lshift", CODE_LSHIFT},
    {"rshift", CODE_RSHIFT},
    // constants
    {"false", CODE_FALSE},
    {"true", CODE_TRUE}
  });

  template <typename T, typename I>
  ForthMachineOf<T, I>::ForthMachineOf(const std::string& source,
                                       int64_t stack_max_depth,
                                       int64_t recursion_max_depth,
                                       int64_t string_buffer_size,
                                       int64_t output_initial_size,
                                       double output_resize_factor)
    : source_(source)
    , output_initial_size_(output_initial_size)
    , output_resize_factor_(output_resize_factor)

    , stack_buffer_(new T[(size_t)stack_max_depth])
    , stack_depth_(0)
    , stack_max_depth_(stack_max_depth)

    , string_buffer_(new char[(size_t)string_buffer_size])
    , string_buffer_size_(string_buffer_size)

    , current_inputs_()
    , current_outputs_()
    , is_ready_(false)

    , current_which_(new int64_t[(size_t)recursion_max_depth])
    , current_where_(new int64_t[(size_t)recursion_max_depth])
    , recursion_current_depth_(0)
    , recursion_max_depth_(recursion_max_depth)

    , do_recursion_depth_(new int64_t[(size_t)recursion_max_depth])
    , do_stop_(new int64_t[(size_t)recursion_max_depth])
    , do_i_(new int64_t[(size_t)recursion_max_depth])
    , do_current_depth_(0)

    , current_error_(util::ForthError::none)

    , count_instructions_(0)
    , count_reads_(0)
    , count_writes_(0)
    , count_nanoseconds_(0)
  {
    std::vector<std::string> tokenized;
    std::vector<std::pair<int64_t, int64_t>> linecol;
    tokenize(tokenized, linecol);
    compile(tokenized, linecol);
  }

  template <typename T, typename I>
  ForthMachineOf<T, I>::~ForthMachineOf() {
    delete [] stack_buffer_;
    delete [] string_buffer_;
    delete [] current_which_;
    delete [] current_where_;
    delete [] do_recursion_depth_;
    delete [] do_stop_;
    delete [] do_i_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::abi_version() const noexcept {
    return ABI_VERSION;
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::source() const noexcept {
    return source_;
  }

  template <typename T, typename I>
  const std::vector<I>
  ForthMachineOf<T, I>::bytecodes() const {
    return bytecodes_;
  }

  template <typename T, typename I>
  const std::vector<int64_t>
  ForthMachineOf<T, I>::bytecodes_offsets() const {
    return bytecodes_offsets_;
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::decompiled() const {
    bool first = true;
    std::stringstream out;

    for (auto const& name : variable_names_) {
      first = false;
      out << "variable " << name << std::endl;
    }

    for (auto const& name : input_names_) {
      first = false;
      out << "input " << name << std::endl;
    }

    for (IndexTypeOf<int64_t> i = 0;  i < output_names_.size();  i++) {
      first = false;
      out << "output " << output_names_[i] << " "
          << util::dtype_to_name(output_dtypes_[i]) << std::endl;
    }

    for (IndexTypeOf<int64_t> i = 0;  i < dictionary_names_.size();  i++) {
      if (!first) {
        out << std::endl;
      }
      first = false;
      int64_t segment_position = dictionary_bytecodes_[i] - BOUND_DICTIONARY;
      out << ": " << dictionary_names_[i] << std::endl
          << (segment_nonempty(segment_position) ? "  " : "")
          << decompiled_segment(segment_position, "  ")
          << ";" << std::endl;
    }

    if (!first  &&  bytecodes_offsets_[1] != 0) {
      out << std::endl;
    }
    out << decompiled_segment(0);
    return std::move(out.str());
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::decompiled_segment(int64_t segment_position,
                                           const std::string& indent,
                                           bool endline) const {
    if (segment_position < 0  ||  (IndexTypeOf<int64_t>)segment_position + 1 >= bytecodes_offsets_.size()) {
      throw std::runtime_error(
        std::string("segment ") + std::to_string(segment_position)
        + std::string(" does not exist in the bytecode") + FILENAME(__LINE__));
    }
    std::stringstream out;
    int64_t bytecode_position = bytecodes_offsets_[(IndexTypeOf<int64_t>)segment_position];
    while (bytecode_position < bytecodes_offsets_[(IndexTypeOf<int64_t>)segment_position + 1]) {
      if (bytecode_position != bytecodes_offsets_[(IndexTypeOf<int64_t>)segment_position]) {
        out << indent;
      }
      out << decompiled_at(bytecode_position, indent);
      bytecode_position += bytecodes_per_instruction(bytecode_position);
      if (endline || bytecode_position < bytecodes_offsets_[(IndexTypeOf<int64_t>)segment_position + 1]) {
        out << std::endl;
      }
    }
    return std::move(out.str());
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::decompiled_at(int64_t bytecode_position,
                                      const std::string& indent) const {
    if (bytecode_position < 0  ||  (IndexTypeOf<int64_t>)bytecode_position >= bytecodes_.size()) {
      throw std::runtime_error(
        std::string("absolute position ") + std::to_string(bytecode_position)
        + std::string(" does not exist in the bytecode") + FILENAME(__LINE__));
    }

    I bytecode = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position];
    I next_bytecode = -1;
    if ((IndexTypeOf<int64_t>)bytecode_position + 1 < bytecodes_.size()) {
      next_bytecode = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
    }

    if (bytecode < 0) {
      I in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
      std::string in_name = input_names_[(IndexTypeOf<int64_t>)in_num];

      std::string rep = (~bytecode & READ_REPEATED) ? "#" : "";
      std::string big = ((~bytecode & READ_BIGENDIAN) != 0) ? "!" : "";
      std::string rest;
      int64_t next_pos = 2;
      I nbits = 0;
      switch (~bytecode & READ_MASK) {
        case READ_BOOL:
          rest = "?->";
          break;
        case READ_INT8:
          rest = "b->";
          break;
        case READ_INT16:
          rest = "h->";
          break;
        case READ_INT32:
          rest = "i->";
          break;
        case READ_INT64:
          rest = "q->";
          break;
        case READ_INTP:
          rest = "n->";
          break;
        case READ_UINT8:
          rest = "B->";
          break;
        case READ_UINT16:
          rest = "H->";
          break;
        case READ_UINT32:
          rest = "I->";
          break;
        case READ_UINT64:
          rest = "Q->";
          break;
        case READ_UINTP:
          rest = "N->";
          break;
        case READ_FLOAT32:
          rest = "f->";
          break;
        case READ_FLOAT64:
          rest = "d->";
          break;
        case READ_VARINT:
          rest = "varint->";
          break;
        case READ_ZIGZAG:
          rest = "zigzag->";
          break;
        case READ_NBIT:
          nbits = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + (IndexTypeOf<int64_t>)next_pos];
          next_pos++;
          rest = std::to_string(nbits) + "bit->";
          break;
        case READ_TEXTINT:
          rest = "textint->";
          break;
        case READ_TEXTFLOAT:
          rest = "textfloat->";
          break;
        case READ_QUOTEDSTR:
          rest = "quotedstr->";
          break;
      }
      std::string arrow = rep + big + rest;

      std::string out_name = "stack";
      if (~bytecode & READ_DIRECT) {
        I out_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + (IndexTypeOf<int64_t>)next_pos];
        out_name = output_names_[(IndexTypeOf<int64_t>)out_num];
      }
      return in_name + std::string(" ") + arrow + std::string(" ") + out_name;
    }

    else if (bytecode >= BOUND_DICTIONARY  &&  next_bytecode == CODE_AGAIN) {
      int64_t body = bytecode - BOUND_DICTIONARY;
      return std::move(std::string("begin\n")
             + (segment_nonempty(body) ? indent + "  " : "")
             + decompiled_segment(body, indent + "  ")
             + indent + "again");
    }

    else if (bytecode >= BOUND_DICTIONARY  &&  next_bytecode == CODE_UNTIL) {
      int64_t body = bytecode - BOUND_DICTIONARY;
      return std::move(std::string("begin\n")
             + (segment_nonempty(body) ? indent + "  " : "")
             + decompiled_segment(body, indent + "  ")
             + indent + "until");
    }

    else if (bytecode >= BOUND_DICTIONARY  &&  next_bytecode == CODE_WHILE) {
      int64_t precondition = bytecode - BOUND_DICTIONARY;
      int64_t postcondition = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 2] - BOUND_DICTIONARY;
      return std::move(std::string("begin\n")
             + (segment_nonempty(precondition) ? indent + "  " : "")
             + decompiled_segment(precondition, indent + "  ")
             + indent + "while\n"
             + (segment_nonempty(postcondition) ? indent + "  " : "")
             + decompiled_segment(postcondition, indent + "  ")
             + indent + "repeat");
    }

    else if (bytecode >= BOUND_DICTIONARY) {
      for (IndexTypeOf<int64_t> i = 0;  i < dictionary_names_.size();  i++) {
        if (dictionary_bytecodes_[i] == bytecode) {
          return dictionary_names_[i];
        }
      }
      return decompiled_segment(bytecode - BOUND_DICTIONARY, indent, false);
    }

    else {
      switch (bytecode) {
        case CODE_LITERAL: {
          return std::to_string(bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1]);
        }
        case CODE_HALT: {
          return "halt";
        }
        case CODE_PAUSE: {
          return "pause";
        }
        case CODE_IF: {
          int64_t consequent = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1] - BOUND_DICTIONARY;
          return std::move(std::string("if\n")
                 + (segment_nonempty(consequent) ? indent + "  " : "")
                 + decompiled_segment(consequent, indent + "  ")
                 + indent + "then");
        }
        case CODE_IF_ELSE: {
          int64_t consequent = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1] - BOUND_DICTIONARY;
          int64_t alternate = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 2] - BOUND_DICTIONARY;
          return std::move(std::string("if\n")
                 + (segment_nonempty(consequent) ? indent + "  " : "")
                 + decompiled_segment(consequent, indent + "  ")
                 + indent + "else\n"
                 + (segment_nonempty(alternate) ? indent + "  " : "")
                 + decompiled_segment(alternate, indent + "  ")
                 + indent + "then");
        }
        case CODE_CASE_REGULAR: {
          int64_t start = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1] - BOUND_DICTIONARY;
          int64_t stop = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 2] - BOUND_DICTIONARY;
          int64_t num_cases = stop - start;
          std::stringstream out;
          out << "case ( regular )\n";
          for (int64_t i = 0;  i < num_cases;  i++) {
            out << indent << "  " << i << " of";
            I consequent = (I)(start + i);
            if (segment_nonempty(consequent)) {
              out << "\n" << indent << "    ";
              out << decompiled_segment(consequent, indent + "    ");
              out << indent << "  endof\n";
            }
            else {
              out << " endof\n";
            }
          }
          bool alt_nonempty = (bytecodes_offsets_[(IndexTypeOf<int64_t>)stop] + 1
                            != bytecodes_offsets_[(IndexTypeOf<int64_t>)stop + 1]);
          if (alt_nonempty) {
            std::string alt_segment = decompiled_segment(stop, indent + "    ");
            // remove the last "drop" command, which is for execution but not visible in source code
            alt_segment = alt_segment.substr(0, alt_segment.length() - indent.length() - 9);
            out << indent << "  ( default )\n" << indent << "    " << alt_segment;
          }
          out << "endcase";
          return std::move(out.str());
        }
        case CODE_DO: {
          int64_t body = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1] - BOUND_DICTIONARY;
          return std::move(std::string("do\n")
                 + (segment_nonempty(body) ? indent + "  " : "")
                 + decompiled_segment(body, indent + "  ")
                 + indent + "loop");
        }
        case CODE_DO_STEP: {
          int64_t body = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1] - BOUND_DICTIONARY;
          return std::move(std::string("do\n")
                 + (segment_nonempty(body) ? indent + "  " : "")
                 + decompiled_segment(body, indent + "  ")
                 + indent + "+loop");
        }
        case CODE_EXIT: {
          return std::move(std::string("exit"));
        }
        case CODE_PUT: {
          int64_t var_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return variable_names_[(IndexTypeOf<int64_t>)var_num] + " !";
        }
        case CODE_INC: {
          int64_t var_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return variable_names_[(IndexTypeOf<int64_t>)var_num] + " +!";
        }
        case CODE_GET: {
          int64_t var_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return variable_names_[(IndexTypeOf<int64_t>)var_num] + " @";
        }
        case CODE_ENUM: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          int64_t start = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 2];
          int64_t stop = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 3];
          std::stringstream out;
          out << input_names_[(IndexTypeOf<int64_t>)in_num] << " enum";
          for (int64_t i = start;  i < stop;  i++) {
            out << " s\" " << strings_[(size_t)i] << "\"";
          }
          return out.str();
        }
        case CODE_ENUMONLY: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          int64_t start = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 2];
          int64_t stop = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 3];
          std::stringstream out;
          out << input_names_[(IndexTypeOf<int64_t>)in_num] << " enumonly";
          for (int64_t i = start;  i < stop;  i++) {
            out << " s\" " << strings_[(size_t)i] << "\"";
          }
          return out.str();
        }
        case CODE_PEEK: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " peek";
        }
        case CODE_LEN_INPUT: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " len";
        }
        case CODE_POS: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " pos";
        }
        case CODE_END: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " end";
        }
        case CODE_SEEK: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " seek";
        }
        case CODE_SKIP: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " skip";
        }
        case CODE_SKIPWS: {
          int64_t in_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return input_names_[(IndexTypeOf<int64_t>)in_num] + " skipws";
        }
        case CODE_WRITE: {
          int64_t out_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return output_names_[(IndexTypeOf<int64_t>)out_num] + " <- stack";
        }
        case CODE_WRITE_ADD: {
          int64_t out_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return output_names_[(IndexTypeOf<int64_t>)out_num] + " +<- stack";
        }
        case CODE_WRITE_DUP: {
          int64_t out_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return output_names_[(IndexTypeOf<int64_t>)out_num] + " dup";
        }
        case CODE_LEN_OUTPUT: {
          int64_t out_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return output_names_[(IndexTypeOf<int64_t>)out_num] + " len";
        }
        case CODE_REWIND: {
          int64_t out_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return output_names_[(IndexTypeOf<int64_t>)out_num] + " rewind";
        }
        case CODE_STRING: {
          int64_t string_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return "s\" " + strings_[(IndexTypeOf<int64_t>)string_num] + "\"";
        }
        case CODE_PRINT_STRING: {
          int64_t string_num = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
          return ".\" " + strings_[(IndexTypeOf<int64_t>)string_num] + "\"";
        }
        case CODE_PRINT: {
          return ".";
        }
        case CODE_PRINT_CR: {
          return "cr";
        }
        case CODE_PRINT_STACK: {
          return ".s";
        }
        case CODE_I: {
          return "i";
        }
        case CODE_J: {
          return "j";
        }
        case CODE_K: {
          return "k";
        }
        case CODE_DUP: {
          return "dup";
        }
        case CODE_DROP: {
          return "drop";
        }
        case CODE_SWAP: {
          return "swap";
        }
        case CODE_OVER: {
          return "over";
        }
        case CODE_ROT: {
          return "rot";
        }
        case CODE_NIP: {
          return "nip";
        }
        case CODE_TUCK: {
          return "tuck";
        }
        case CODE_ADD: {
          return "+";
        }
        case CODE_SUB: {
          return "-";
        }
        case CODE_MUL: {
          return "*";
        }
        case CODE_DIV: {
          return "/";
        }
        case CODE_MOD: {
          return "mod";
        }
        case CODE_DIVMOD: {
          return "/mod";
        }
        case CODE_NEGATE: {
          return "negate";
        }
        case CODE_ADD1: {
          return "1+";
        }
        case CODE_SUB1: {
          return "1-";
        }
        case CODE_ABS: {
          return "abs";
        }
        case CODE_MIN: {
          return "min";
        }
        case CODE_MAX: {
          return "max";
        }
        case CODE_EQ: {
          return "=";
        }
        case CODE_NE: {
          return "<>";
        }
        case CODE_GT: {
          return ">";
        }
        case CODE_GE: {
          return ">=";
        }
        case CODE_LT: {
          return "<";
        }
        case CODE_LE: {
          return "<=";
        }
        case CODE_EQ0: {
          return "0=";
        }
        case CODE_INVERT: {
          return "invert";
        }
        case CODE_AND: {
          return "and";
        }
        case CODE_OR: {
          return "or";
        }
        case CODE_XOR: {
          return "xor";
        }
        case CODE_LSHIFT: {
          return "lshift";
        }
        case CODE_RSHIFT: {
          return "rshift";
        }
        case CODE_FALSE: {
          return "false";
        }
        case CODE_TRUE: {
          return "true";
        }
      }
      return std::move(std::string("(unrecognized bytecode ") + std::to_string(bytecode) + ")");
    }
  }

  template <typename T, typename I>
  const std::vector<std::string>
  ForthMachineOf<T, I>::dictionary() const {
    return dictionary_names_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::stack_max_depth() const noexcept {
    return stack_max_depth_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::recursion_max_depth() const noexcept {
    return recursion_max_depth_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::string_buffer_size() const noexcept {
    return string_buffer_size_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::output_initial_size() const noexcept {
    return output_initial_size_;
  }

  template <typename T, typename I>
  double
  ForthMachineOf<T, I>::output_resize_factor() const noexcept {
    return output_resize_factor_;
  }

  template <typename T, typename I>
  const std::vector<T>
  ForthMachineOf<T, I>::stack() const {
    std::vector<T> out;
    for (int64_t i = 0;  i < stack_depth_;  i++) {
      out.push_back(stack_buffer_[i]);
    }
    return out;
  }

  template <typename T, typename I>
  T
  ForthMachineOf<T, I>::stack_at(int64_t from_top) const noexcept {
    return stack_buffer_[stack_depth_ - from_top];
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::stack_depth() const noexcept {
    return stack_depth_;
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::stack_clear() noexcept {
    stack_depth_ = 0;
  }

  template <typename T, typename I>
  const std::map<std::string, T>
  ForthMachineOf<T, I>::variables() const {
    std::map<std::string, T> out;
    for (IndexTypeOf<int64_t> i = 0;  i < variable_names_.size();  i++) {
      out[variable_names_[i]] = variables_[i];
    }
    return out;
  }

  template <typename T, typename I>
  const std::vector<std::string>
  ForthMachineOf<T, I>::variable_index() const {
    return variable_names_;
  }

  template <typename T, typename I>
  T
  ForthMachineOf<T, I>::variable_at(const std::string& name) const {
    for (IndexTypeOf<int64_t> i = 0;  i < variable_names_.size();  i++) {
      if (variable_names_[i] == name) {
        return variables_[i];
      }
    }
    throw std::invalid_argument(
      std::string("variable not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  T
  ForthMachineOf<T, I>::variable_at(int64_t index) const noexcept {
    return variables_[(IndexTypeOf<int64_t>)index];
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::input_must_be_writable(const std::string& name) const {
    for (IndexTypeOf<int64_t> i = 0;  i < input_names_.size();  i++) {
      if (input_names_[i] == name) {
        return input_must_be_writable_[i];
      }
    }
    throw std::invalid_argument(
      std::string("input not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::input_position_at(const std::string& name) const {
    for (IndexTypeOf<int64_t> i = 0;
         i < input_names_.size()  &&  i < current_inputs_.size();
         i++) {
      if (input_names_[i] == name) {
        return current_inputs_[i].get()->pos();
      }
    }
    throw std::invalid_argument(
      std::string("input not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::input_position_at(int64_t index) const noexcept {
    return current_inputs_[(IndexTypeOf<int64_t>)index].get()->pos();
  }

  template <typename T, typename I>
  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>
  ForthMachineOf<T, I>::outputs() const {
    std::map<std::string, std::shared_ptr<ForthOutputBuffer>> out;
    for (IndexTypeOf<int64_t> i = 0;
         i < output_names_.size()  &&  i < current_outputs_.size();
         i++) {
      out[output_names_[i]] = current_outputs_[i];
    }
    return out;
  }

  template <typename T, typename I>
  const std::vector<std::string>
  ForthMachineOf<T, I>::output_index() const noexcept {
    return output_names_;
  }

  template <typename T, typename I>
  const std::shared_ptr<ForthOutputBuffer>
  ForthMachineOf<T, I>::output_at(const std::string& name) const {
    for (IndexTypeOf<int64_t> i = 0;
         i < output_names_.size()  &&  i < current_outputs_.size();
         i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i];
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const std::shared_ptr<ForthOutputBuffer>
  ForthMachineOf<T, I>::output_at(int64_t index) const noexcept {
    return current_outputs_[(IndexTypeOf<int64_t>)index];
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::string_at(int64_t index) const noexcept {
    return ((index >= 0  &&  index < (int64_t)strings_.size()) ?
      strings_[(IndexTypeOf<int64_t>)index] : std::string("a string at ")
      + std::to_string(index) + std::string(" is undefined"));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::reset() {
    stack_depth_ = 0;
    for (IndexTypeOf<int64_t> i = 0;  i < variables_.size();  i++) {
      variables_[i] = 0;
    }
    current_inputs_.clear();
    current_outputs_.clear();
    is_ready_ = false;
    recursion_current_depth_ = 0;
    while (!recursion_target_depth_.empty()) {
      recursion_target_depth_.pop();
    }
    do_current_depth_ = 0;
    current_error_ = util::ForthError::none;
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::begin(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {
    reset();
    current_inputs_ = std::vector<std::shared_ptr<ForthInputBuffer>>();
    for (auto name : input_names_) {
      bool found = false;
      for (auto pair : inputs) {
        if (pair.first == name) {
          current_inputs_.push_back(pair.second);
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::invalid_argument(
          std::string("AwkwardForth source code defines an input that was not provided: ")
          + name + FILENAME(__LINE__)
        );
      }
    }

    current_outputs_ = std::vector<std::shared_ptr<ForthOutputBuffer>>();
    int64_t init = output_initial_size_;
    double resize = output_resize_factor_;
    for (IndexTypeOf<int64_t> i = 0;  i < output_names_.size();  i++) {
      std::shared_ptr<ForthOutputBuffer> out;
      switch (output_dtypes_[i]) {
        case util::dtype::boolean: {
          out = std::make_shared<ForthOutputBufferOf<bool>>(init, resize);
          break;
        }
        case util::dtype::int8: {
          out = std::make_shared<ForthOutputBufferOf<int8_t>>(init, resize);
          break;
        }
        case util::dtype::int16: {
          out = std::make_shared<ForthOutputBufferOf<int16_t>>(init, resize);
          break;
        }
        case util::dtype::int32: {
          out = std::make_shared<ForthOutputBufferOf<int32_t>>(init, resize);
          break;
        }
        case util::dtype::int64: {
          out = std::make_shared<ForthOutputBufferOf<int64_t>>(init, resize);
          break;
        }
        case util::dtype::uint8: {
          out = std::make_shared<ForthOutputBufferOf<uint8_t>>(init, resize);
          break;
        }
        case util::dtype::uint16: {
          out = std::make_shared<ForthOutputBufferOf<uint16_t>>(init, resize);
          break;
        }
        case util::dtype::uint32: {
          out = std::make_shared<ForthOutputBufferOf<uint32_t>>(init, resize);
          break;
        }
        case util::dtype::uint64: {
          out = std::make_shared<ForthOutputBufferOf<uint64_t>>(init, resize);
          break;
        }
        case util::dtype::float32: {
          out = std::make_shared<ForthOutputBufferOf<float>>(init, resize);
          break;
        }
        case util::dtype::float64: {
          out = std::make_shared<ForthOutputBufferOf<double>>(init, resize);
          break;
        }
        default: {
          throw std::runtime_error(std::string("unhandled ForthOutputBuffer type")
                                   + FILENAME(__LINE__));
        }
      }
      current_outputs_.push_back(out);
    }

    recursion_target_depth_.push(0);
    bytecodes_pointer_push(0);
    is_ready_ = true;
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::begin() {
    const std::map<std::string, std::shared_ptr<ForthInputBuffer>> inputs;
    begin(inputs);
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::step() {
    if (!is_ready()) {
      current_error_ = util::ForthError::not_ready;
      return current_error_;
    }
    if (is_done()) {
      current_error_ = util::ForthError::is_done;
      return current_error_;
    }
    if (current_error_ != util::ForthError::none) {
      return current_error_;
    }

    int64_t recursion_target_depth_top = recursion_target_depth_.top();

    auto begin_time = std::chrono::high_resolution_clock::now();
    internal_run(true, recursion_target_depth_top);
    auto end_time = std::chrono::high_resolution_clock::now();

    count_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - begin_time
    ).count();

    if (recursion_current_depth_ == recursion_target_depth_.top()) {
      recursion_target_depth_.pop();
    }

    return current_error_;
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::begin_again(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs, bool reset_instruction) {
    if (!is_ready()) {
      throw std::invalid_argument(
          std::string("'begin' not called on the AwkwardForth machine, 'begin_again' invalid")
          + FILENAME(__LINE__)
        );
    }
    if (current_error_ != util::ForthError::none) {
      return current_error_;
    }

    current_inputs_ = std::vector<std::shared_ptr<ForthInputBuffer>>();
    for (auto name : input_names_) {
      bool found = false;
      for (auto pair : inputs) {
        if (pair.first == name) {
          current_inputs_.push_back(pair.second);
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::invalid_argument(
          std::string("AwkwardForth source code defines an input that was not provided: ")
          + name + FILENAME(__LINE__)
        );
      }
    }
  if (reset_instruction){
    recursion_target_depth_.push(0);
    bytecodes_pointer_push(0);
  }
  return current_error_;
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {
    begin(inputs);

    int64_t recursion_target_depth_top = recursion_target_depth_.top();

    auto begin_time = std::chrono::high_resolution_clock::now();
    internal_run(false, recursion_target_depth_top);
    auto end_time = std::chrono::high_resolution_clock::now();

    count_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - begin_time
    ).count();

    if (recursion_current_depth_ == recursion_target_depth_.top()) {
      recursion_target_depth_.pop();
    }

    return current_error_;
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::run() {
    const std::map<std::string, std::shared_ptr<ForthInputBuffer>> inputs;
    return run(inputs);
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::resume() {
    if (!is_ready()) {
      current_error_ = util::ForthError::not_ready;
      return current_error_;
    }
    if (is_done()) {
      current_error_ = util::ForthError::is_done;
      return current_error_;
    }
    if (current_error_ != util::ForthError::none) {
      return current_error_;
    }

    int64_t recursion_target_depth_top = recursion_target_depth_.top();

    auto begin_time = std::chrono::high_resolution_clock::now();
    internal_run(false, recursion_target_depth_top);
    auto end_time = std::chrono::high_resolution_clock::now();

    count_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - begin_time
    ).count();

    if (recursion_current_depth_ == recursion_target_depth_.top()) {
      recursion_target_depth_.pop();
    }

    return current_error_;
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::call(const std::string& name) {
    for (IndexTypeOf<int64_t> i = 0;  i < dictionary_names_.size();  i++) {
      if (dictionary_names_[i] == name) {
        return call((int64_t)i);
      }
    }
    throw std::runtime_error(
      std::string("AwkwardForth unrecognized word: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::call(int64_t index) {
    if (!is_ready()) {
      current_error_ = util::ForthError::not_ready;
      return current_error_;
    }
    if (current_error_ != util::ForthError::none) {
      return current_error_;
    }

    recursion_target_depth_.push(recursion_current_depth_);
    bytecodes_pointer_push(dictionary_bytecodes_[(IndexTypeOf<int64_t>)index] - BOUND_DICTIONARY);

    int64_t recursion_target_depth_top = recursion_target_depth_.top();

    auto begin_time = std::chrono::high_resolution_clock::now();
    internal_run(false, recursion_target_depth_top);
    auto end_time = std::chrono::high_resolution_clock::now();

    count_nanoseconds_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - begin_time
    ).count();

    if (recursion_current_depth_ == recursion_target_depth_.top()) {
      recursion_target_depth_.pop();
    }

    return current_error_;
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::maybe_throw(util::ForthError /* err */,  // FIXME: this argument is not needed
                                    const std::set<util::ForthError>& ignore) const {
    if (ignore.count(current_error_) == 0) {
      switch (current_error_) {
        case util::ForthError::not_ready: {
          throw std::invalid_argument(
            "'not ready' in AwkwardForth runtime: call 'begin' before 'step' or "
            "'resume' (note: check 'is_ready')");
        }
        case util::ForthError::is_done: {
          throw std::invalid_argument(
            "'is done' in AwkwardForth runtime: reached the end of the program; "
            "call 'begin' to 'step' again (note: check 'is_done')");
        }
        case util::ForthError::user_halt: {
          throw std::invalid_argument(
            "'user halt' in AwkwardForth runtime: user-defined error or stopping "
            "condition");
        }
        case util::ForthError::recursion_depth_exceeded: {
          throw std::invalid_argument(
            "'recursion depth exceeded' in AwkwardForth runtime: too many words "
            "calling words or a recursive word is looping endlessly");
        }
        case util::ForthError::stack_underflow: {
          throw std::invalid_argument(
            "'stack underflow' in AwkwardForth runtime: tried to pop from an empty "
            "stack");
        }
        case util::ForthError::stack_overflow: {
          throw std::invalid_argument(
            "'stack overflow' in AwkwardForth runtime: tried to push beyond the "
            "predefined maximum stack depth");
        }
        case util::ForthError::read_beyond: {
          throw std::invalid_argument(
            "'read beyond' in AwkwardForth runtime: tried to read beyond the end "
            "of an input");
        }
        case util::ForthError::seek_beyond: {
          throw std::invalid_argument(
            "'seek beyond' in AwkwardForth runtime: tried to seek beyond the bounds "
            "of an input (0 or length)");
        }
        case util::ForthError::skip_beyond: {
          throw std::invalid_argument(
            "'skip beyond' in AwkwardForth runtime: tried to skip beyond the bounds "
            "of an input (0 or length)");
        }
        case util::ForthError::rewind_beyond: {
          throw std::invalid_argument(
            "'rewind beyond' in AwkwardForth runtime: tried to rewind beyond the "
            "beginning of an output");
        }
        case util::ForthError::division_by_zero: {
          throw std::invalid_argument(
            "'division by zero' in AwkwardForth runtime: tried to divide by zero");
        }
        case util::ForthError::varint_too_big: {
          throw std::invalid_argument(
            "'varint too big' in AwkwardForth runtime: variable-length integer is "
            "too big to represent as a fixed-width integer");
        }
        case util::ForthError::text_number_missing: {
          throw std::invalid_argument(
            "'text number missing' in AwkwardForth runtime: expected a number in "
            "input text, didn't find one");
        }
        case util::ForthError::quoted_string_missing: {
          throw std::invalid_argument(
            "'quoted string missing' in AwkwardForth runtime: expected a quoted string in "
            "input text, didn't find one");
        }
        case util::ForthError::enumeration_missing: {
          throw std::invalid_argument(
            "'enumeration missing' in AwkwardForth runtime: expected one of several "
            "enumerated values in the input text, didn't find one");
        }
        default:
          break;
      }
    }
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::current_bytecode_position() const noexcept {
    if (recursion_current_depth_ == 0) {
      return -1;
    }
    else {
      int64_t which = current_which_[recursion_current_depth_ - 1];
      int64_t where = current_where_[recursion_current_depth_ - 1];
      if (where < bytecodes_offsets_[(IndexTypeOf<int64_t>)which + 1] - bytecodes_offsets_[(IndexTypeOf<int64_t>)which]) {
        return bytecodes_offsets_[(IndexTypeOf<int64_t>)which] + where;
      }
      else {
        return -1;
      }
    }
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::current_recursion_depth() const noexcept {
    if (recursion_target_depth_.empty()) {
      return -1;
    }
    else {
      return recursion_current_depth_ - recursion_target_depth_.top();
    }
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::current_instruction() const {
    int64_t bytecode_position = current_bytecode_position();
    if (bytecode_position == -1) {
      throw std::invalid_argument(
        "'is done' in AwkwardForth runtime: reached the end of the program or segment; "
        "call 'begin' to 'step' again (note: check 'is_done')"
        + FILENAME(__LINE__)
      );
    }
    else {
      return decompiled_at(bytecode_position, "");
    }
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::count_reset() noexcept {
    count_instructions_ = 0;
    count_reads_ = 0;
    count_writes_ = 0;
    count_nanoseconds_ = 0;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_instructions() const noexcept {
    return count_instructions_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_reads() const noexcept {
    return count_reads_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_writes() const noexcept {
    return count_writes_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_nanoseconds() const noexcept {
    return count_nanoseconds_;
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_integer(const std::string& word, int64_t& value) const {
    if (word.size() >= 2  &&  word.substr(0, 2) == std::string("0x")) {
      try {
        value = (int64_t)std::stoul(word.substr(2, word.size() - 2), nullptr, 16);
      }
      catch (std::invalid_argument& err) {
        return false;
      }
      return true;
    }
    else {
      try {
        value = (int64_t)std::stoul(word, nullptr, 10);
      }
      catch (std::invalid_argument& err) {
        return false;
      }
      return true;
    }
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_variable(const std::string& word) const {
    return std::find(variable_names_.begin(),
                     variable_names_.end(), word) != variable_names_.end();
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_input(const std::string& word) const {
    return std::find(input_names_.begin(),
                     input_names_.end(), word) != input_names_.end();
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_output(const std::string& word) const {
    return std::find(output_names_.begin(),
                     output_names_.end(), word) != output_names_.end();
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_nbit(const std::string& word, I& value) const {
    std::string parser = word;
    if (parser.length() != 0  &&  parser[0] == '#') {
      parser = parser.substr(1, parser.length() - 1);
    }
    if (parser.length() != 0  &&  parser[0] == '!') {
      parser = parser.substr(1, parser.length() - 1);
    }
    if (parser.length() > 5  &&  parser.substr(parser.length() - 5, 5) == "bit->") {
      std::string number = parser.substr(0, parser.length() - 5);
      try {
        value = std::stoi(number, nullptr, 10);
      }
      catch (std::invalid_argument& err) {
        return false;
      }
      if (0 < value  &&  value <= 64) {
        return true;
      }
      else {
        value = 0;
        return false;
      }
    }
    else {
      return false;
    }
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_reserved(const std::string& word) const {
    I num;
    if (is_nbit(word, num)) {
      return true;
    }
    else {
      return reserved_words_.find(word) != reserved_words_.end()  ||
             input_parser_words_.find(word) != input_parser_words_.end()  ||
             output_dtype_words_.find(word) != output_dtype_words_.end()  ||
             generic_builtin_words_.find(word) != generic_builtin_words_.end();
    }
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_defined(const std::string& word) const {
    for (auto name : dictionary_names_) {
      if (name == word) {
        return true;
      }
    }
    return false;
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::segment_nonempty(int64_t segment_position) const {
    return bytecodes_offsets_[(IndexTypeOf<int64_t>)segment_position] != bytecodes_offsets_[(IndexTypeOf<int64_t>)segment_position + 1];
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::bytecodes_per_instruction(int64_t bytecode_position) const {
    I bytecode = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position];
    I next_bytecode = -1;
    if ((IndexTypeOf<int64_t>)bytecode_position + 1 < bytecodes_.size()) {
      next_bytecode = bytecodes_[(IndexTypeOf<int64_t>)bytecode_position + 1];
    }

    if (bytecode < 0) {
      int64_t total = 2;
      if ((~bytecode & READ_MASK) == READ_NBIT) {
        total++;
      }
      if (~bytecode & READ_DIRECT) {
        total++;
      }
      return total;
    }
    else if (bytecode >= BOUND_DICTIONARY  &&
             (next_bytecode == CODE_AGAIN  ||  next_bytecode == CODE_UNTIL)) {
      return 2;
    }
    else if (bytecode >= BOUND_DICTIONARY  &&
             (next_bytecode == CODE_WHILE)) {
      return 3;
    }
    else {
      switch (bytecode) {
        case CODE_ENUM:
        case CODE_ENUMONLY:
          return 4;
        case CODE_IF_ELSE:
        case CODE_CASE_REGULAR:
          return 3;
        case CODE_LITERAL:
        case CODE_IF:
        case CODE_DO:
        case CODE_DO_STEP:
        case CODE_EXIT:
        case CODE_PUT:
        case CODE_INC:
        case CODE_GET:
        case CODE_PEEK:
        case CODE_LEN_INPUT:
        case CODE_POS:
        case CODE_END:
        case CODE_SEEK:
        case CODE_SKIP:
        case CODE_SKIPWS:
        case CODE_WRITE:
        case CODE_WRITE_ADD:
        case CODE_WRITE_DUP:
        case CODE_LEN_OUTPUT:
        case CODE_REWIND:
        case CODE_STRING:
        case CODE_PRINT_STRING:
          return 2;
        default:
          return 1;
      }
    }
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::err_linecol(const std::vector<std::pair<int64_t, int64_t>>& linecol,
                                    int64_t startpos,
                                    int64_t stoppos,
                                    const std::string& message) const {
    std::pair<int64_t, int64_t> lc = linecol[(IndexTypeOf<int64_t>)startpos];
    std::stringstream out;
    out << "in AwkwardForth source code, line " << lc.first << " col " << lc.second
        << ", " << message << ":" << std::endl << std::endl << "    ";
    int64_t line = 1;
    int64_t col = 1;
    IndexTypeOf<int64_t> start = 0;
    IndexTypeOf<int64_t> stop = 0;
    while (stop < source_.length()) {
      if (lc.first == line  &&  lc.second == col) {
        start = stop;
      }
      if ((IndexTypeOf<int64_t>)stoppos < linecol.size()  &&
          linecol[(IndexTypeOf<int64_t>)stoppos].first == line  &&  linecol[(IndexTypeOf<int64_t>)stoppos].second == col) {
        break;
      }
      if (source_[stop] == '\n') {
        line++;
        col = 0;
      }
      col++;
      stop++;
    }
    out << source_.substr(start, stop - start);
    return std::move(out.str());
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::tokenize(std::vector<std::string>& tokenized,
                                 std::vector<std::pair<int64_t, int64_t>>& linecol) {
    IndexTypeOf<int64_t> start = 0;
    IndexTypeOf<int64_t> stop = 0;
    bool full = false;
    int64_t line = 1;
    int64_t colstart = 0;
    int64_t colstop = 0;
    while (stop < source_.size()) {
      char current = source_[stop];
      // Whitespace separates tokens and is not included in them.
      if (current == ' '  ||  current == '\r'  ||  current == '\t'  ||
          current == '\v'  ||  current == '\f') {
        if (full) {
          tokenized.push_back(source_.substr(start, stop - start));
          linecol.push_back(std::pair<int64_t, int64_t>(line, colstart));
        }
        start = stop;
        full = false;
        colstart = colstop;
      }
      // '\n' is considered a token because it terminates '\\ .. \n' comments.
      // It has no semantic meaning after the parsing stage.
      else if (current == '\n') {
        if (full) {
          tokenized.push_back(source_.substr(start, stop - start));
          linecol.push_back(std::pair<int64_t, int64_t>(line, colstart));
        }
        tokenized.push_back(source_.substr(stop, 1));
        linecol.push_back(std::pair<int64_t, int64_t>(line, colstart));
        start = stop;
        full = false;
        line++;
        colstart = 0;
        colstop = 0;
      }
      // Everything else is part of a token (Forth word).
      else {
        if (!full) {
          start = stop;
          colstart = colstop;
        }
        full = true;
      }
      stop++;
      colstop++;

      if (!tokenized.empty()  &&  (tokenized[tokenized.size() - 1] == ".\""
        ||  tokenized[tokenized.size() - 1] == "s\"")) {
        // Strings are tokenized differently.
        if (stop == source_.size()) {
          throw std::invalid_argument(
            std::string("unclosed string after .\" or s\" word") + FILENAME(__LINE__));
        }
        int64_t nextline = line;
        current = source_[stop];
        start = stop;
        colstart = colstop;
        while (current != '\"'  ||  source_[stop - 1] == '\\') {
          if (current == '\n') {
            nextline++;
            colstart = 0;
            colstop = 0;
          }
          stop++;
          colstop++;
          if (stop == source_.size()) {
            throw std::invalid_argument(
              std::string("unclosed string after .\" or s\" word") + FILENAME(__LINE__));
          }
          current = source_[stop];
        }
        stop++;
        colstop++;
        std::string str = source_.substr(start, stop - start - 1);
        size_t pos = 0;
        while ((pos = str.find("\\\"", pos)) != std::string::npos) {
          str.replace(pos, 2, "\"");
          pos++;
        }
        tokenized.push_back(str);
        linecol.push_back(std::pair<int64_t, int64_t>(line, colstart));
        start = stop;
        full = false;
        colstart = colstop;
      }
    }
    // The source code might end on non-whitespace.
    if (full) {
      tokenized.push_back(source_.substr(start, stop - start));
      linecol.push_back(std::pair<int64_t, int64_t>(line, colstart));
    }
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::compile(const std::vector<std::string>& tokenized,
                                const std::vector<std::pair<int64_t, int64_t>>& linecol) {
    std::vector<std::vector<I>> dictionary;

    // Start recursive parsing.
    std::vector<I> bytecodes;
    dictionary.push_back(bytecodes);
    parse("",
          tokenized,
          linecol,
          0,
          (int64_t)tokenized.size(),
          bytecodes,
          dictionary,
          0,
          0);
    dictionary[0] = bytecodes;

    // Copy std::vector<std::vector<I>> to flattened contents and offsets.
    bytecodes_offsets_.push_back(0);
    for (auto segment : dictionary) {
      for (auto bytecode : segment) {
        bytecodes_.push_back(bytecode);
      }
      bytecodes_offsets_.push_back((int64_t)bytecodes_.size());
    }
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::parse(const std::string& defn,
                              const std::vector<std::string>& tokenized,
                              const std::vector<std::pair<int64_t, int64_t>>& linecol,
                              int64_t start,
                              int64_t stop,
                              std::vector<I>& bytecodes,
                              std::vector<std::vector<I>>& dictionary,
                              int64_t exitdepth,
                              int64_t dodepth) {
    int64_t pos = start;
    while (pos < stop) {
      std::string word = tokenized[(IndexTypeOf<std::string>)pos];

      if (word == "(") {
        // Simply skip the parenthesized text: it's a comment.
        int64_t substop = pos;
        int64_t nesting = 1;
        while (nesting > 0) {
          substop++;
          if (substop >= stop) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, substop, "'(' is missing its closing ')'")
              + FILENAME(__LINE__)
            );
          }
          // Any parentheses in the comment text itself must be balanced.
          if (tokenized[(IndexTypeOf<std::string>)substop] == "(") {
            nesting++;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == ")") {
            nesting--;
          }
        }

        pos = substop + 1;
      }

      else if (word == "\\") {
        // Modern, backslash-to-end-of-line comments. Nothing needs to be balanced.
        int64_t substop = pos;
        while (substop < stop  &&  tokenized[(IndexTypeOf<std::string>)substop] != "\n") {
          substop++;
        }

        pos = substop + 1;
      }

      else if (word == "\n") {
        // This is a do-nothing token to delimit backslash-to-end-of-line comments.
        pos++;
      }

      else if (word == "") {
        // Just in case there's a leading or trailing blank in the token stream.
        pos++;
      }

      else if (word == ":") {
        if (pos + 1 >= stop  ||  tokenized[(IndexTypeOf<std::string>)pos + 1] == ";") {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 2, "missing name in word definition")
              + FILENAME(__LINE__)
            );
        }
        std::string name = tokenized[(IndexTypeOf<std::string>)pos + 1];
        int64_t num;

        if (is_input(name)  ||  is_output(name)  ||  is_variable(name)  ||
            is_defined(name)  ||  is_reserved(name)  ||  is_integer(name, num)) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2,
                        "input names, output names, variable names, and user-defined "
                        "words must all be unique and not reserved words or integers")
            + FILENAME(__LINE__)
          );
        }

        int64_t substart = pos + 2;
        int64_t substop = pos + 1;
        int64_t nesting = 1;
        while (nesting > 0) {
          substop++;
          if (substop >= stop) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, stop,
                          "definition is missing its closing ';'")
              + FILENAME(__LINE__)
            );
          }
          if (tokenized[(IndexTypeOf<std::string>)substop] == ":") {
            nesting++;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == ";") {
            nesting--;
          }
        }

        // Add the new word to the dictionary before parsing it so that recursive
        // functions can be defined.
        I bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
        dictionary_names_.push_back(name);
        dictionary_bytecodes_.push_back(bytecode);

        // Now parse the subroutine and add it to the dictionary.
        std::vector<I> body;
        dictionary.push_back(body);
        parse(name,
              tokenized,
              linecol,
              substart,
              substop,
              body,
              dictionary,
              0,
              0);
        dictionary[(IndexTypeOf<I>)bytecode - BOUND_DICTIONARY] = body;

        pos = substop + 1;
      }

      else if (word == "recurse") {
        if (defn == "") {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 1,
                        "only allowed in a ': name ... ;' definition")
              + FILENAME(__LINE__)
          );
        }
        for (IndexTypeOf<I> i = 0;  i < dictionary_names_.size();  i++) {
          if (dictionary_names_[i] == defn) {
            bytecodes.push_back(dictionary_bytecodes_[i]);
          }
        }

        pos++;
      }

      else if (word == "variable") {
        if (pos + 1 >= stop) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2,
                        "missing name in variable declaration")
            + FILENAME(__LINE__)
          );
        }
        std::string name = tokenized[(IndexTypeOf<std::string>)pos + 1];

        int64_t num;


        if (is_input(name)  ||  is_output(name)  ||  is_variable(name)  ||
            is_defined(name)  ||  is_reserved(name)  ||  is_integer(name, num)) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2,
                        "input names, output names, variable names, and user-defined "
                        "words must all be unique and not reserved words or integers")
            + FILENAME(__LINE__)
          );
        }

        variable_names_.push_back(name);
        variables_.push_back(0);

        pos += 2;
      }

      else if (word == "input") {
        if (pos + 1 >= stop) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2, "missing name in input declaration")
            + FILENAME(__LINE__)
          );
        }
        std::string name = tokenized[(IndexTypeOf<std::string>)pos + 1];

        int64_t num;
        if (is_input(name)  ||  is_output(name)  ||  is_variable(name)  ||
            is_defined(name)  ||  is_reserved(name)  ||  is_integer(name, num)) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2,
                        "input names, output names, variable names, and user-defined "
                        "words must all be unique and not reserved words or integers")
            + FILENAME(__LINE__)
          );
        }

        input_names_.push_back(name);
        input_must_be_writable_.push_back(false);

        pos += 2;
      }

      else if (word == "output") {
        if (pos + 2 >= stop) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 3,
                        "missing name or dtype in output declaration")
            + FILENAME(__LINE__)
          );
        }
        std::string name = tokenized[(IndexTypeOf<std::string>)pos + 1];
        std::string dtype_string = tokenized[(IndexTypeOf<std::string>)pos + 2];

        int64_t num;
        if (is_input(name)  ||  is_output(name)  ||  is_variable(name)  ||
            is_defined(name)  ||  is_reserved(name)  ||  is_integer(name, num)) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2,
                        "input names, output names, variable names, and user-defined "
                        "words must all be unique and not reserved words or integers")
            + FILENAME(__LINE__)
          );
        }

        bool found_dtype = false;
        for (auto pair : output_dtype_words_) {
          if (pair.first == dtype_string) {
            output_names_.push_back(name);
            output_dtypes_.push_back(pair.second);
            found_dtype = true;
            break;
          }
        }
        if (!found_dtype) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 3, "output dtype not recognized")
            + FILENAME(__LINE__)
          );
        }

        pos += 3;
      }

      else if (word == "halt") {
        bytecodes.push_back(CODE_HALT);

        pos++;
      }

      else if (word == "pause") {
        bytecodes.push_back(CODE_PAUSE);

        pos++;
      }

      else if (word == "if") {
        int64_t substart = pos + 1;
        int64_t subelse = -1;
        int64_t substop = pos;
        int64_t nesting = 1;
        while (nesting > 0) {
          substop++;
          if (substop >= stop) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, stop, "'if' is missing its closing 'then'")
              + FILENAME(__LINE__)
            );
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "if") {
            nesting++;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "then") {
            nesting--;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "else"  &&  nesting == 1) {
            subelse = substop;
          }
        }

        if (subelse == -1) {
          // Add the consequent to the dictionary so that it can be used
          // without special instruction pointer manipulation at runtime.
          I bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> consequent;
          dictionary.push_back(consequent);
          parse(defn,
                tokenized,
                linecol,
                substart,
                substop,
                consequent,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode - BOUND_DICTIONARY] = consequent;

          bytecodes.push_back(CODE_IF);
          bytecodes.push_back(bytecode);

          pos = substop + 1;
        }
        else {
          // Same as above, except that two new definitions must be made.
          I bytecode1 = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> consequent;
          dictionary.push_back(consequent);
          parse(defn,
                tokenized,
                linecol,
                substart,
                subelse,
                consequent,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode1 - BOUND_DICTIONARY] = consequent;

          I bytecode2 = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> alternate;
          dictionary.push_back(alternate);
          parse(defn,
                tokenized,
                linecol,
                subelse + 1,
                substop,
                alternate,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode2 - BOUND_DICTIONARY] = alternate;

          bytecodes.push_back(CODE_IF_ELSE);
          bytecodes.push_back(bytecode1);
          bytecodes.push_back(bytecode2);

          pos = substop + 1;
        }
      }

      else if (word == "case") {
        std::vector<int64_t> ofs;
        std::vector<int64_t> endofs;
        int64_t substop = pos;
        int64_t nesting = 1;
        while (nesting > 0) {
          substop++;
          if (substop >= stop) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, stop, "'case' is missing its closing 'endcase'")
              + FILENAME(__LINE__)
            );
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "case") {
            nesting++;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "endcase") {
            nesting--;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "of"  &&  nesting == 1) {
            ofs.push_back(substop);
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "endof"  &&  nesting == 1) {
            endofs.push_back(substop);
          }
        }

        // check the lengths and orders of 'of' and 'endof'
        if (ofs.size() != endofs.size()) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, stop, "in 'case' .. 'endcase', there must be an 'endof' for every 'of'")
            + FILENAME(__LINE__)
          );
        }
        for (int64_t i = 0;  (size_t)i < ofs.size();  i++) {
          if (ofs[(size_t)i] > endofs[(size_t)i]) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, stop, "in 'case' .. 'endcase', there must be an 'endof' for every 'of'")
            + FILENAME(__LINE__)
          );
          }
        }

        std::vector<I> predicates;
        std::vector<I> consequents;
        I alternate;

        I first_bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
        for(I i = 0;  (size_t)i < ofs.size() + 1;  i++){
          dictionary.push_back({});
        }
        bool can_specialize = true;
        int64_t substart = pos + 1;
        for (int64_t i = 0;  (size_t)i < ofs.size();  i++) {
          I pred_bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> pred;
          dictionary.push_back(pred);
          parse(defn,
                tokenized,
                linecol,
                substart,
                ofs[(size_t)i],
                pred,
                dictionary,
                exitdepth + 1,
                dodepth);
          if (pred != std::vector<I>({ CODE_LITERAL, (int32_t)i })) {
            can_specialize = false;
          }
          dictionary[(IndexTypeOf<int64_t>)pred_bytecode - BOUND_DICTIONARY] = pred;
          predicates.push_back(pred_bytecode);

          I cons_bytecode = first_bytecode + (I)i;
          std::vector<I> cons;
          parse(defn,
                tokenized,
                linecol,
                ofs[(size_t)i] + 1,
                endofs[(size_t)i],
                cons,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)cons_bytecode - BOUND_DICTIONARY] = cons;
          consequents.push_back(cons_bytecode);

          substart = endofs[(size_t)i] + 1;
        }

        {
          I alt_bytecode = first_bytecode + (I)ofs.size();
          std::vector<I> alt;
          parse(defn,
                tokenized,
                linecol,
                substart,
                substop,
                alt,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)alt_bytecode - BOUND_DICTIONARY] = alt;
          alternate = alt_bytecode;
        }

        if (can_specialize) {
          // Specialized 'case' statement can be turned into a jump table:
          //
          // CASE                         CODE_CASE_REGULAR
          // 0 OF ... ENDOF               first dictionary index (predicate)
          // 1 OF ... ENDOF               dictionary index of default case (last)
          // 2 OF ... ENDOF
          // ... ( default case )
          // ENDCASE
          //
          // where CODE_CASE_REGULAR pops an item off the stack; if it's nonnegative
          // and less than n, the dictionary index is computed from it by adding
          // the "dictionary index of case 0"; if it's out of range, the dictionary
          // index is for the default case.

          auto alt = dictionary.begin() + (alternate - BOUND_DICTIONARY);
          alt->push_back(CODE_DROP);  // append "drop"

          bytecodes.push_back(CODE_CASE_REGULAR);
          bytecodes.push_back(first_bytecode);
          bytecodes.push_back(alternate);
        }

        else {
          // General 'case' statement can be turned into 'if' chain:
          //
          // CASE
          // test1 OF ... ENDOF           test1 OVER = IF DROP ... ELSE
          // test2 OF ... ENDOF           test2 OVER = IF DROP ... ELSE
          // testn OF ... ENDOF           testn OVER = IF DROP ... ELSE
          // ... ( default case )         ...
          // ENDCASE                      DROP THEN [THEN [THEN ...]]
          //
          // But the regular case should become a jump table with CODE_CASE_REGULAR.

          for (int64_t i = 0;  (size_t)i < ofs.size();  i++) {
            auto pred = dictionary.begin() + (predicates[(size_t)i] - BOUND_DICTIONARY);
            pred->push_back(CODE_OVER);  // append "over"
            pred->push_back(CODE_EQ);    // append "="
            auto cons = dictionary.begin() + (consequents[(size_t)i] - BOUND_DICTIONARY);
            cons->insert(cons->begin(), CODE_DROP);  // prepend "drop"
          }
          auto alt = dictionary.begin() + (alternate - BOUND_DICTIONARY);
          alt->push_back(CODE_DROP);  // append "drop"

          I bytecode2 = alternate;
          for (int64_t i = (int64_t)ofs.size() - 1;  i >= 0;  i--) {
            I bytecode1 = consequents[(size_t)i];

            I ifthenelse_bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
            std::vector<I> ifthenelse;
            dictionary.push_back(ifthenelse);
            ifthenelse.push_back(predicates[(size_t)i]);
            ifthenelse.push_back(CODE_IF_ELSE);
            ifthenelse.push_back(bytecode1);
            ifthenelse.push_back(bytecode2);
            dictionary[(IndexTypeOf<int64_t>)ifthenelse_bytecode - BOUND_DICTIONARY] = ifthenelse;

            bytecode2 = ifthenelse_bytecode;
          }

          bytecodes.push_back(bytecode2);
        }

        pos = substop + 1;
      }

      else if (word == "do") {
        int64_t substart = pos + 1;
        int64_t substop = pos;
        bool is_step = false;
        int64_t nesting = 1;
        while (nesting > 0) {
          substop++;
          if (substop >= stop) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, stop,
                          "'do' is missing its closing 'loop'")
              + FILENAME(__LINE__)
            );
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "do") {
            nesting++;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "loop") {
            nesting--;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "+loop") {
            if (nesting == 1) {
              is_step = true;
            }
            nesting--;
          }
        }

        // Add the loop body to the dictionary so that it can be used
        // without special instruction pointer manipulation at runtime.
        I bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
        std::vector<I> body;
        dictionary.push_back(body);
        parse(defn,
              tokenized,
              linecol,
              substart,
              substop,
              body,
              dictionary,
              exitdepth + 1,
              dodepth + 1);
        dictionary[(IndexTypeOf<int64_t>)bytecode - BOUND_DICTIONARY] = body;

        if (is_step) {
          bytecodes.push_back(CODE_DO_STEP);
          bytecodes.push_back(bytecode);
        }
        else {
          bytecodes.push_back(CODE_DO);
          bytecodes.push_back(bytecode);
        }

        pos = substop + 1;
      }

      else if (word == "begin") {
        int64_t substart = pos + 1;
        int64_t substop = pos;
        bool is_again = false;
        int64_t subwhile = -1;
        int64_t nesting = 1;
        while (nesting > 0) {
          substop++;
          if (substop >= stop) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, stop,
                          "'begin' is missing its closing 'until' or 'while ... repeat'")
              + FILENAME(__LINE__)
            );
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "begin") {
            nesting++;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "until") {
            nesting--;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "again") {
            if (nesting == 1) {
              is_again = true;
            }
            nesting--;
          }
          else if (tokenized[(IndexTypeOf<std::string>)substop] == "while") {
            if (nesting == 1) {
              subwhile = substop;
            }
            nesting--;
            int64_t subnesting = 1;
            while (subnesting > 0) {
              substop++;
              if (substop >= stop) {
                throw std::invalid_argument(
                  err_linecol(linecol, pos, stop,
                              "'while' is missing its closing 'repeat'")
                  + FILENAME(__LINE__)
                );
              }
              else if (tokenized[(IndexTypeOf<std::string>)substop] == "while") {
                subnesting++;
              }
              else if (tokenized[(IndexTypeOf<std::string>)substop] == "repeat") {
                subnesting--;
              }
            }
          }
        }

        if (is_again) {
          // Add the 'begin ... again' body to the dictionary so that it can be
          // used without special instruction pointer manipulation at runtime.
          I bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> body;
          dictionary.push_back(body);
          parse(defn,
                tokenized,
                linecol,
                substart,
                substop,
                body,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode - BOUND_DICTIONARY] = body;

          bytecodes.push_back(bytecode);
          bytecodes.push_back(CODE_AGAIN);

          pos = substop + 1;
        }
        else if (subwhile == -1) {
          // Same for the 'begin .. until' body.
          I bytecode = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> body;
          dictionary.push_back(body);
          parse(defn,
                tokenized,
                linecol,
                substart,
                substop,
                body,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode - BOUND_DICTIONARY] = body;

          bytecodes.push_back(bytecode);
          bytecodes.push_back(CODE_UNTIL);

          pos = substop + 1;
        }
        else {
          // Same for the 'begin .. repeat' statements.
          I bytecode1 = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> precondition;
          dictionary.push_back(precondition);
          parse(defn,
                tokenized,
                linecol,
                substart,
                subwhile,
                precondition,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode1 - BOUND_DICTIONARY] = precondition;

          // Same for the 'repeat .. until' statements.
          I bytecode2 = (I)dictionary.size() + BOUND_DICTIONARY;
          std::vector<I> postcondition;
          dictionary.push_back(postcondition);
          parse(defn,
                tokenized,
                linecol,
                subwhile + 1,
                substop,
                postcondition,
                dictionary,
                exitdepth + 1,
                dodepth);
          dictionary[(IndexTypeOf<int64_t>)bytecode2 - BOUND_DICTIONARY] = postcondition;

          bytecodes.push_back(bytecode1);
          bytecodes.push_back(CODE_WHILE);
          bytecodes.push_back(bytecode2);

          pos = substop + 1;
        }
      }

      else if (word == "exit") {
        bytecodes.push_back(CODE_EXIT);
        bytecodes.push_back((int32_t)exitdepth);

        pos++;
      }

      else if (is_variable(word)) {
        IndexTypeOf<int64_t> variable_index = 0;
        for (;  variable_index < variable_names_.size();  variable_index++) {
          if (variable_names_[variable_index] == word) {
            break;
          }
        }
        if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "!") {
          bytecodes.push_back(CODE_PUT);
          bytecodes.push_back((int32_t)variable_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "+!") {
          bytecodes.push_back(CODE_INC);
          bytecodes.push_back((int32_t)variable_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "@") {
          bytecodes.push_back(CODE_GET);
          bytecodes.push_back((int32_t)variable_index);

          pos += 2;
        }
        else {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2, "missing '!', '+!', or '@' "
                        "after variable name")
          );
        }
      }

      else if (is_input(word)) {
        IndexTypeOf<I> input_index = 0;
        for (;  input_index < input_names_.size();  input_index++) {
          if (input_names_[input_index] == word) {
            break;
          }
        }

        if (pos + 1 < stop  &&  (tokenized[(IndexTypeOf<std::string>)pos + 1] == "enum"  ||
                                 tokenized[(IndexTypeOf<std::string>)pos + 1] == "enumonly")) {
          if (tokenized[(IndexTypeOf<std::string>)pos + 1] == "enum") {
            bytecodes.push_back(CODE_ENUM);
          }
          else {
            bytecodes.push_back(CODE_ENUMONLY);
          }
          bytecodes.push_back((int32_t)input_index);
          bytecodes.push_back((int32_t)strings_.size());
          size_t start_size = strings_.size();

          if (pos + 2 >= stop) {
            if (tokenized[(IndexTypeOf<std::string>)pos + 1] == "enum") {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 2, "need at least one string (s\" word) after \"enum\"")
                + FILENAME(__LINE__)
              );
            }
            else {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 2, "need at least one string (s\" word) after \"enumonly\"")
                + FILENAME(__LINE__)
              );
            }
          }

          pos += 2;

          while (pos < stop) {
            std::string next_word = tokenized[(IndexTypeOf<std::string>)pos];
            if (next_word == "s\"") {
              if (pos + 1 >= stop) {
                throw std::invalid_argument(
                  err_linecol(linecol, pos, pos + 2, "unclosed string after s\" word")
                  + FILENAME(__LINE__)
                );
              }
              strings_.push_back(tokenized[(IndexTypeOf<std::string>)pos + 1]);

              pos += 2;
            }
            else {
              break;
            }
          }

          if (strings_.size() == start_size) {
            if (tokenized[(IndexTypeOf<std::string>)pos + 1] == "enum") {
              throw std::invalid_argument(
                err_linecol(linecol, pos - 2, pos + 1, "need at least one string (s\" word) after \"enum\"")
                + FILENAME(__LINE__)
              );
            }
            else {
              throw std::invalid_argument(
                err_linecol(linecol, pos - 2, pos + 1, "need at least one string (s\" word) after \"enumonly\"")
                + FILENAME(__LINE__)
              );
            }
          }

          bytecodes.push_back((int32_t)strings_.size());
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "peek") {
          bytecodes.push_back(CODE_PEEK);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "len") {
          bytecodes.push_back(CODE_LEN_INPUT);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "pos") {
          bytecodes.push_back(CODE_POS);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "end") {
          bytecodes.push_back(CODE_END);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "seek") {
          bytecodes.push_back(CODE_SEEK);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "skip") {
          bytecodes.push_back(CODE_SKIP);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "skipws") {
          bytecodes.push_back(CODE_SKIPWS);
          bytecodes.push_back((int32_t)input_index);

          pos += 2;
        }
        else if (pos + 1 < stop) {
          I bytecode = 0;

          std::string parser = tokenized[(IndexTypeOf<std::string>)pos + 1];

          if (parser.length() != 0  &&  parser[0] == '#') {
            bytecode |= READ_REPEATED;
            parser = parser.substr(1, parser.length() - 1);
          }

          if (parser.length() != 0  &&  parser[0] == '!') {
            bytecode |= READ_BIGENDIAN;
            parser = parser.substr(1, parser.length() - 1);
          }

          bool must_be_writable = ((bytecode & READ_REPEATED) != 0);
          if (NATIVELY_BIG_ENDIAN) {
            must_be_writable &= ((bytecode & READ_BIGENDIAN) == 0);
          }
          else {
            must_be_writable &= ((bytecode & READ_BIGENDIAN) != 0);
          }
          input_must_be_writable_[input_index] = must_be_writable;

          bool good = true;
          I nbits = 0;
          if (parser.length() != 0) {
            if (parser == "varint->") {
              bytecode |= READ_VARINT;
              parser = parser.substr(parser.length() - 2, 2);
            }
            else if (parser == "zigzag->") {
              bytecode |= READ_ZIGZAG;
              parser = parser.substr(parser.length() - 2, 2);
            }
            else if (is_nbit(parser, nbits)) {
              bytecode |= READ_NBIT;
              parser = parser.substr(parser.length() - 2, 2);
            }
            else if (parser == "textint->") {
              bytecode |= READ_TEXTINT;
              parser = parser.substr(parser.length() - 2, 2);
            }
            else if (parser == "textfloat->") {
              bytecode |= READ_TEXTFLOAT;
              parser = parser.substr(parser.length() - 2, 2);
            }
            else if (parser == "quotedstr->") {
              bytecode |= READ_QUOTEDSTR;
              parser = parser.substr(parser.length() - 2, 2);
            }
            else {
              switch (parser[0]) {
                case '?': {
                  bytecode |= READ_BOOL;
                  break;
                }
                case 'b': {
                  bytecode |= READ_INT8;
                  break;
                }
                case 'h': {
                  bytecode |= READ_INT16;
                  break;
                }
                case 'i': {
                   bytecode |= READ_INT32;
                   break;
                 }
                case 'q': {
                   bytecode |= READ_INT64;
                   break;
                 }
                case 'n': {
                  bytecode |= READ_INTP;
                  break;
                }
                case 'B': {
                  bytecode |= READ_UINT8;
                  break;
                }
                case 'H': {
                  bytecode |= READ_UINT16;
                  break;
                }
                case 'I': {
                  bytecode |= READ_UINT32;
                  break;
                }
                case 'Q': {
                  bytecode |= READ_UINT64;
                  break;
                }
                case 'N': {
                  bytecode |= READ_UINTP;
                  break;
                }
                case 'f': {
                  bytecode |= READ_FLOAT32;
                  break;
                }
                case 'd': {
                  bytecode |= READ_FLOAT64;
                  break;
                }
                default: {
                  good = false;
                }
              }
              if (good) {
                parser = parser.substr(1, parser.length() - 1);
              }
            }
          }

          if (!good  ||  parser != "->") {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 3,
                          "missing '*-> stack/output', "
                          "'seek', 'skip', 'skipws', 'end', 'pos', or 'len' after input name")
              + FILENAME(__LINE__)
            );
          }

          bool found_output = false;
          IndexTypeOf<I> output_index = 0;
          if (pos + 2 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 2] == "stack") {
            // not READ_DIRECT
            if ((bytecode & READ_MASK) == READ_TEXTFLOAT) {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 3,
                            "'stack' not allowed after 'textfloat->'")
                + FILENAME(__LINE__)
              );
            }
            if ((bytecode & READ_MASK) == READ_QUOTEDSTR) {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 3,
                            "'stack' not allowed after 'quotedstr->'")
                + FILENAME(__LINE__)
              );
            }
          }
          else if (pos + 2 < stop  &&  is_output(tokenized[(IndexTypeOf<std::string>)pos + 2])) {
            for (;  output_index < output_names_.size();  output_index++) {
              if (output_names_[output_index] == tokenized[(IndexTypeOf<std::string>)pos + 2]) {
                found_output = true;
                break;
              }
            }
            bytecode |= READ_DIRECT;
          }
          else {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 3,
                          "missing 'stack' or 'output' after '*->'")
              + FILENAME(__LINE__)
            );
          }

          // Parser instructions are bit-flipped to detect them by the sign bit.
          bytecodes.push_back(~bytecode);
          bytecodes.push_back((int32_t)input_index);
          if (nbits > 0) {
            bytecodes.push_back((int32_t)nbits);
          }
          if (found_output) {
            bytecodes.push_back((int32_t)output_index);
          }

          pos += 3;
        }
        else {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 3,
                        "missing '*-> stack/output', 'seek', 'skip', 'skipws', 'end', "
                        "'pos', or 'len' after input name")
            + FILENAME(__LINE__)
          );
        }
      }

      else if (is_output(word)) {
        IndexTypeOf<I> output_index = 0;
        for (;  output_index < output_names_.size();  output_index++) {
          if (output_names_[output_index] == word) {
            break;
          }
        }
        if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "<-") {
          if (pos + 2 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 2] == "stack") {
            bytecodes.push_back(CODE_WRITE);
            bytecodes.push_back((int32_t)output_index);

            pos += 3;
          }
          else {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 3, "missing 'stack' after '<-'")
              + FILENAME(__LINE__)
            );
          }
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "+<-") {
          if (pos + 2 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 2] == "stack") {
            bytecodes.push_back(CODE_WRITE_ADD);
            bytecodes.push_back((int32_t)output_index);

            pos += 3;
          }
          else {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 3, "missing 'stack' after '+<-'")
              + FILENAME(__LINE__)
            );
          }
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "dup") {
          bytecodes.push_back(CODE_WRITE_DUP);
          bytecodes.push_back((int32_t)output_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "len") {
          bytecodes.push_back(CODE_LEN_OUTPUT);
          bytecodes.push_back((int32_t)output_index);

          pos += 2;
        }
        else if (pos + 1 < stop  &&  tokenized[(IndexTypeOf<std::string>)pos + 1] == "rewind") {
          bytecodes.push_back(CODE_REWIND);
          bytecodes.push_back((int32_t)output_index);

          pos += 2;
        }
        else {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2, "missing '<- stack', '+<- stack', "
                        "'dup', 'len', or 'rewind' after output name")
            + FILENAME(__LINE__)
          );
        }
      }

      else if (word == "s\"") {
        bytecodes.push_back(CODE_STRING);
        bytecodes.push_back((int32_t)strings_.size());

        if (pos + 1 >= stop) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2, "unclosed string after s\" word")
            + FILENAME(__LINE__)
          );
        }
        strings_.push_back(tokenized[(IndexTypeOf<std::string>)pos + 1]);

        pos += 2;
      }

      else if (word == ".\"") {
        bytecodes.push_back(CODE_PRINT_STRING);
        bytecodes.push_back((int32_t)strings_.size());

        if (pos + 1 >= stop) {
          throw std::invalid_argument(
            err_linecol(linecol, pos, pos + 2, "unclosed string after .\" word")
            + FILENAME(__LINE__)
          );
        }
        strings_.push_back(tokenized[(IndexTypeOf<std::string>)pos + 1]);

        pos += 2;
      }

      else {
        bool found_in_builtins = false;
        for (auto pair : generic_builtin_words_) {
          if (pair.first == word) {
            found_in_builtins = true;
            if (word == "i"  &&  dodepth < 1) {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 1, "only allowed in a 'do' loop")
                + FILENAME(__LINE__)
              );
            }
            if (word == "j"  &&  dodepth < 2) {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 1, "only allowed in a nested 'do' loop")
                + FILENAME(__LINE__)
              );
            }
            if (word == "k"  &&  dodepth < 3) {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 1, "only allowed in a doubly nested 'do' loop")
                + FILENAME(__LINE__)
              );
            }
            bytecodes.push_back((int32_t)pair.second);

            pos++;
          }
        }

        if (!found_in_builtins) {
          bool found_in_dictionary = false;
          for (IndexTypeOf<std::string> i = 0;  i < dictionary_names_.size();  i++) {
            if (dictionary_names_[i] == word) {
              found_in_dictionary = true;
              bytecodes.push_back((int32_t)dictionary_bytecodes_[i]);

              pos++;
            }
          }

          if (!found_in_dictionary) {
            int64_t num;
            if (is_integer(word, num)) {
              bytecodes.push_back(CODE_LITERAL);
              bytecodes.push_back((int32_t)num);

              pos++;
            }

            else {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 1, "unrecognized word or wrong context for word")
                + FILENAME(__LINE__)
              );
            }
          } // !found_in_dictionary
        } // !found_in_builtins
      } // generic instruction
    } // end loop over segment
  }

  // For bit-flipping: https://stackoverflow.com/a/2603254/1623645
  static uint8_t bitswap_lookup[16] = {0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
                                       0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::internal_run(bool single_step, int64_t recursion_target_depth_top) { // noexcept
    while (recursion_current_depth_ != recursion_target_depth_top) {
      while (bytecodes_pointer_where() < (
                 bytecodes_offsets_[(IndexTypeOf<int64_t>)bytecodes_pointer_which() + 1] -
                 bytecodes_offsets_[(IndexTypeOf<int64_t>)bytecodes_pointer_which()]
             )) {
        I bytecode = bytecode_get();

        if (do_current_depth_ == 0  ||
            do_abs_recursion_depth() != recursion_current_depth_) {
          // Normal operation: step forward one bytecode.
          bytecodes_pointer_where()++;
        }
        else if (do_i() >= do_stop()) {
          // End a 'do' loop.
          do_current_depth_--;
          bytecodes_pointer_where()++;
          continue;
        }
        // else... don't increase bytecode_pointer_where()

        if (bytecode < 0) {
          bool byteswap;
          if (NATIVELY_BIG_ENDIAN) {
            byteswap = ((~bytecode & READ_BIGENDIAN) == 0);
          }
          else {
            byteswap = ((~bytecode & READ_BIGENDIAN) != 0);
          }

          I in_num = bytecode_get();
          bytecodes_pointer_where()++;

          int64_t num_items = 1;
          if (~bytecode & READ_REPEATED) {
            if (stack_cannot_pop()) {
              current_error_ = util::ForthError::stack_underflow;
              return;
            }
            num_items = stack_pop();
          }

          I format = ~bytecode & READ_MASK;

          if (format == READ_VARINT) {
            ForthInputBuffer* input = current_inputs_[(IndexTypeOf<int64_t>)in_num].get();
            ForthOutputBuffer* output = nullptr;
            if (~bytecode & READ_DIRECT) {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              output = current_outputs_[(IndexTypeOf<int64_t>)out_num].get();
            }

            if (output == nullptr) {
              for (int64_t count = 0;  count < num_items;  count++) {
                uint64_t result = input->read_varint(current_error_);
                if (current_error_ != util::ForthError::none) {
                  return;
                }
                if (stack_cannot_push()) {
                  current_error_ = util::ForthError::stack_overflow;
                  return;
                }
                stack_push((T)result);   // note: pushing result
              }
            }
            else {
              for (int64_t count = 0;  count < num_items;  count++) {
                uint64_t result = input->read_varint(current_error_);
                if (current_error_ != util::ForthError::none) {
                  return;
                }
                output->write_one_uint64(result, false);   // note: writing result as unsigned
              }
            }
          }

          else if (format == READ_ZIGZAG) {
            ForthInputBuffer* input = current_inputs_[(IndexTypeOf<int64_t>)in_num].get();
            ForthOutputBuffer* output = nullptr;
            if (~bytecode & READ_DIRECT) {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              output = current_outputs_[(IndexTypeOf<int64_t>)out_num].get();
            }

            if (output == nullptr) {
              for (int64_t count = 0;  count < num_items;  count++) {
                int64_t result = input->read_zigzag(current_error_);
                if (current_error_ != util::ForthError::none) {
                  return;
                }
                if (stack_cannot_push()) {
                  current_error_ = util::ForthError::stack_overflow;
                  return;
                }
                stack_push((T)result);   // note: pushing value
              }
            }
            else {
              for (int64_t count = 0;  count < num_items;  count++) {
                int64_t result = input->read_zigzag(current_error_);
                if (current_error_ != util::ForthError::none) {
                  return;
                }
                output->write_one_int64(result, false);   // note: writing value as signed
              }
            }
          }

          else if (format == READ_NBIT) {
            // For bit-flipping: https://stackoverflow.com/a/2603254/1623645
            bool flip = (~bytecode & READ_BIGENDIAN) != 0;

            I bit_width = bytecode_get();
            bytecodes_pointer_where()++;

            ForthInputBuffer* input = current_inputs_[(IndexTypeOf<int64_t>)in_num].get();
            ForthOutputBuffer* output = nullptr;
            if (~bytecode & READ_DIRECT) {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              output = current_outputs_[(IndexTypeOf<int64_t>)out_num].get();
            }

            uint64_t mask = (1 << bit_width) - 1;
            uint64_t bits_wnd_l = 8;
            uint64_t bits_wnd_r = 0;
            int64_t items_remaining = num_items;
            uint64_t data;
            uint64_t tmp;
            uint8_t tmpbyte;

            if (items_remaining != 0) {
              tmpbyte = input->read_byte(current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }
              tmp = (uint64_t)tmpbyte;
              if (flip) {
                // For bit-flipping: https://stackoverflow.com/a/2603254/1623645
                tmp = (uint64_t)(bitswap_lookup[tmp & 0b1111] << 4) | bitswap_lookup[tmp >> 4];
              }
              data = tmp;
            }
            while (items_remaining != 0) {
              if (bits_wnd_r >= 8) {
                bits_wnd_r -= 8;
                bits_wnd_l -= 8;
                data >>= 8;
              }
              else if (bits_wnd_l - bits_wnd_r >= (uint64_t)bit_width) {
                tmp = (data >> bits_wnd_r) & mask;
                if (output == nullptr) {
                  if (stack_cannot_push()) {
                    current_error_ = util::ForthError::stack_overflow;
                    return;
                  }
                  stack_push((T)tmp);   // note: pushing value
                }
                else {
                  output->write_one_uint64(tmp, false);   // note: writing value as unsigned
                }
                items_remaining--;
                bits_wnd_r += (uint64_t)bit_width;
              }
              else {
                tmpbyte = input->read_byte(current_error_);
                if (current_error_ != util::ForthError::none) {
                  return;
                }
                tmp = (uint64_t)tmpbyte;
                if (flip) {
                  // For bit-flipping: https://stackoverflow.com/a/2603254/1623645
                  tmp = (uint64_t)(bitswap_lookup[tmp & 0b1111] << 4) | bitswap_lookup[tmp >> 4];
                }
                data |= tmp << bits_wnd_l;
                bits_wnd_l += 8;
              }
            }
          }

          else if (format == READ_TEXTINT) {
            ForthInputBuffer* input = current_inputs_[(IndexTypeOf<int64_t>)in_num].get();
            ForthOutputBuffer* output = nullptr;
            if (~bytecode & READ_DIRECT) {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              output = current_outputs_[(IndexTypeOf<int64_t>)out_num].get();
            }

            if (output == nullptr) {
              for (int64_t count = 0;  count < num_items;  count++) {
                if (count != 0) {
                  input->skipws();
                }
                int64_t result = input->read_textint(current_error_);

                if (current_error_ != util::ForthError::none) {
                  return;
                }
                if (stack_cannot_push()) {
                  current_error_ = util::ForthError::stack_overflow;
                  return;
                }
                stack_push((T)result);   // note: pushing result
              }
            }
            else {
              for (int64_t count = 0;  count < num_items;  count++) {
                if (count != 0) {
                  input->skipws();
                }
                int64_t result = input->read_textint(current_error_);
                if (current_error_ != util::ForthError::none) {
                  return;
                }
                output->write_one_int64(result, false);   // note: writing value as signed
              }
            }
          }

          else if (format == READ_TEXTFLOAT) {
            ForthInputBuffer* input = current_inputs_[(IndexTypeOf<int64_t>)in_num].get();
            I out_num = bytecode_get();
            bytecodes_pointer_where()++;
            ForthOutputBuffer* output = current_outputs_[(IndexTypeOf<int64_t>)out_num].get();

            for (int64_t count = 0;  count < num_items;  count++) {
              if (count != 0) {
                input->skipws();
              }
              double result = input->read_textfloat(current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }
              output->write_one_float64(result, false);   // note: writing value as floating-point
            }
          }

          else if (format == READ_QUOTEDSTR) {
            ForthInputBuffer* input = current_inputs_[(IndexTypeOf<int64_t>)in_num].get();
            I out_num = bytecode_get();
            bytecodes_pointer_where()++;
            ForthOutputBuffer* output = current_outputs_[(IndexTypeOf<int64_t>)out_num].get();

            for (int64_t count = 0;  count < num_items;  count++) {
              if (count != 0) {
                input->skipws();
              }
              int64_t length;
              input->read_quotedstr(string_buffer_, string_buffer_size_, length, current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)length);   // note: pushing length
              if (length != 0) {
                output->write_one_string(string_buffer_, length);   // note: copying string
              }
            }
          }

          else if (~bytecode & READ_DIRECT) {
            I out_num = bytecode_get();
            bytecodes_pointer_where()++;

            #define WRITE_DIRECTLY(TYPE, SUFFIX) {                             \
                TYPE* ptr = reinterpret_cast<TYPE*>(                           \
                    current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->read( \
                      num_items * (int64_t)sizeof(TYPE), current_error_));     \
                if (current_error_ != util::ForthError::none) {                \
                  return;                                                      \
                }                                                              \
                if (num_items == 1) {                                          \
                  current_outputs_[(IndexTypeOf<int64_t>)out_num].get()->write_one_##SUFFIX(\
                      *ptr, byteswap);                                         \
                }                                                              \
                else {                                                         \
                  current_outputs_[(IndexTypeOf<int64_t>)out_num].get()->write_##SUFFIX(   \
                      num_items, ptr, byteswap);                               \
                }                                                              \
                break;                                                         \
              }

            switch (format) {
              case READ_BOOL:    WRITE_DIRECTLY(bool, bool)
              case READ_INT8:    WRITE_DIRECTLY(int8_t, int8)
              case READ_INT16:   WRITE_DIRECTLY(int16_t, int16)
              case READ_INT32:   WRITE_DIRECTLY(int32_t, int32)
              case READ_INT64:   WRITE_DIRECTLY(int64_t, int64)
              case READ_INTP:    WRITE_DIRECTLY(ssize_t, intp)
              case READ_UINT8:   WRITE_DIRECTLY(uint8_t, uint8)
              case READ_UINT16:  WRITE_DIRECTLY(uint16_t, uint16)
              case READ_UINT32:  WRITE_DIRECTLY(uint32_t, uint32)
              case READ_UINT64:  WRITE_DIRECTLY(uint64_t, uint64)
              case READ_UINTP:   WRITE_DIRECTLY(size_t, uintp)
              case READ_FLOAT32: WRITE_DIRECTLY(float, float32)
              case READ_FLOAT64: WRITE_DIRECTLY(double, float64)
            }

            count_writes_++;

          } // end if READ_DIRECT

          else {
              # define WRITE_TO_STACK(TYPE) {                                  \
                TYPE* ptr = reinterpret_cast<TYPE*>(                           \
                    current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->read( \
                        num_items * (int64_t)sizeof(TYPE), current_error_));   \
                if (current_error_ != util::ForthError::none) {                \
                  return;                                                      \
                }                                                              \
                for (int64_t i = 0;  i < num_items;  i++) {                    \
                  TYPE value = ptr[i];                                         \
                  if (stack_cannot_push()) {                                   \
                    current_error_ = util::ForthError::stack_overflow;         \
                    return;                                                    \
                  }                                                            \
                  stack_push((T)value);                                        \
                }                                                              \
                break;                                                         \
              }

              # define WRITE_TO_STACK_SWAP(TYPE, SWAP) {                       \
                TYPE* ptr = reinterpret_cast<TYPE*>(                           \
                    current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->read( \
                        num_items * (int64_t)sizeof(TYPE), current_error_));   \
                if (current_error_ != util::ForthError::none) {                \
                  return;                                                      \
                }                                                              \
                for (int64_t i = 0;  i < num_items;  i++) {                    \
                  TYPE value = ptr[i];                                         \
                  if (byteswap) {                                              \
                    SWAP(1, value);                                            \
                  }                                                            \
                  if (stack_cannot_push()) {                                   \
                    current_error_ = util::ForthError::stack_overflow;         \
                    return;                                                    \
                  }                                                            \
                  stack_push((T)value);                                        \
                }                                                              \
                break;                                                         \
              }

              # define WRITE_TO_STACK_SWAP_INTP(TYPE) {                        \
                TYPE* ptr = reinterpret_cast<TYPE*>(                           \
                    current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->read( \
                        num_items * (int64_t)sizeof(TYPE), current_error_));   \
                if (current_error_ != util::ForthError::none) {                \
                  return;                                                      \
                }                                                              \
                for (int64_t i = 0;  i < num_items;  i++) {                    \
                  TYPE value = ptr[i];                                         \
                  if (byteswap) {                                              \
                    byteswap_intp(1, value);                                   \
                  }                                                            \
                  if (stack_cannot_push()) {                                   \
                    current_error_ = util::ForthError::stack_overflow;         \
                    return;                                                    \
                  }                                                            \
                  stack_push((T)value);                                        \
                }                                                              \
                break;                                                         \
              }

            switch (format) {
              case READ_BOOL:    WRITE_TO_STACK(bool)
              case READ_INT8:    WRITE_TO_STACK(int8_t)
              case READ_INT16:   WRITE_TO_STACK_SWAP(int16_t, byteswap16)
              case READ_INT32:   WRITE_TO_STACK_SWAP(int32_t, byteswap32)
              case READ_INT64:   WRITE_TO_STACK_SWAP(int64_t, byteswap64)
              case READ_INTP:    WRITE_TO_STACK_SWAP_INTP(ssize_t)
              case READ_UINT8:   WRITE_TO_STACK(uint8_t)
              case READ_UINT16:  WRITE_TO_STACK_SWAP(uint16_t, byteswap16)
              case READ_UINT32:  WRITE_TO_STACK_SWAP(uint32_t, byteswap32)
              case READ_UINT64:  WRITE_TO_STACK_SWAP(uint64_t, byteswap64)
              case READ_UINTP:   WRITE_TO_STACK_SWAP_INTP(size_t)
              case READ_FLOAT32: WRITE_TO_STACK_SWAP(float, byteswap32)
              case READ_FLOAT64: WRITE_TO_STACK_SWAP(double, byteswap64)
            }

          } // end if not READ_DIRECT (i.e. read to stack)

          count_reads_++;

        } // end if bytecode < 0

        else if (bytecode >= BOUND_DICTIONARY) {
          if (recursion_current_depth_ == recursion_max_depth_) {
            current_error_ = util::ForthError::recursion_depth_exceeded;
            return;
          }
          bytecodes_pointer_push(bytecode - BOUND_DICTIONARY);
        }

        else {
          switch (bytecode) {
            case CODE_LITERAL: {
              I num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)num);
              break;
            }

            case CODE_HALT: {
              is_ready_ = false;
              recursion_current_depth_ = 0;
              while (recursion_target_depth_.size() > 1) {
                recursion_target_depth_.pop();
              }
              do_current_depth_ = 0;
              current_error_ = util::ForthError::user_halt;

              // HALT counts as an instruction.
              count_instructions_++;
              return;
            }

            case CODE_PAUSE: {
              // In case of 'do ... pause loop/+loop', update the do-stack.
              if (is_segment_done()) {
                bytecodes_pointer_pop();

                if (do_current_depth_ != 0  &&
                    do_abs_recursion_depth() == recursion_current_depth_) {
                  // End one step of a 'do ... loop' or a 'do ... +loop'.
                  if (do_loop_is_step()) {
                    if (stack_cannot_pop()) {
                      current_error_ = util::ForthError::stack_underflow;
                      return;
                    }
                    do_i() += stack_pop();
                  }
                  else {
                    do_i()++;
                  }
                }
              }

              // PAUSE counts as an instruction.
              count_instructions_++;
              return;
            }

            case CODE_IF: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_pop() == 0) {
                // Predicate is false, so skip over the next instruction.
                bytecodes_pointer_where()++;
              }
              break;
            }

            case CODE_IF_ELSE: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_pop() == 0) {
                // Predicate is false, so skip over the next instruction
                // but do the one after that.
                bytecodes_pointer_where()++;
              }
              else {
                // Predicate is true, so do the next instruction (we know it's
                // in the dictionary), but skip the one after that.
                I consequent = bytecode_get();
                bytecodes_pointer_where() += 2;
                if (recursion_current_depth_ == recursion_max_depth_) {
                  current_error_ = util::ForthError::recursion_depth_exceeded;
                  return;
                }
                bytecodes_pointer_push(consequent - BOUND_DICTIONARY);

                // Ordinarily, a redirection like the above would count as one.
                count_instructions_++;
              }
              break;
            }

            case CODE_CASE_REGULAR: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* value = stack_peek();

              I start = bytecode_get();
              bytecodes_pointer_where()++;
              I stop = bytecode_get();
              bytecodes_pointer_where()++;

              I which;
              if (*value < 0  ||  *value >= (stop - start)) {
                which = stop;
              }
              else {
                stack_depth_--;
                which = start + (I)(*value);
              }

              if (recursion_current_depth_ == recursion_max_depth_) {
                current_error_ = util::ForthError::recursion_depth_exceeded;
                return;
              }
              bytecodes_pointer_push(which - BOUND_DICTIONARY);

              // Ordinarily, a redirection like the above would count as one.
              count_instructions_++;

              break;
            }

            case CODE_DO: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2();
              if (do_current_depth_ == recursion_max_depth_) {
                current_error_ = util::ForthError::recursion_depth_exceeded;
                return;
              }
              do_loop_push(pair[1], pair[0]);

              break;
            }

            case CODE_DO_STEP: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2();
              if (do_current_depth_ == recursion_max_depth_) {
                current_error_ = util::ForthError::recursion_depth_exceeded;
                return;
              }
              do_steploop_push(pair[1], pair[0]);
              break;
            }

            case CODE_AGAIN: {
              // Go back and do the body again.
              bytecodes_pointer_where() -= 2;
              break;
            }

            case CODE_UNTIL: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_pop() == 0) {
                // Predicate is false, so go back and do the body again.
                bytecodes_pointer_where() -= 2;
              }
              break;
            }

            case CODE_WHILE: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_pop() == 0) {
                // Predicate is false, so skip over the conditional body.
                bytecodes_pointer_where()++;
              }
              else {
                // Predicate is true, so do the next instruction (we know it's
                // in the dictionary), but skip back after that.
                I posttest = bytecode_get();
                bytecodes_pointer_where() -= 2;
                if (recursion_current_depth_ == recursion_max_depth_) {
                  current_error_ = util::ForthError::recursion_depth_exceeded;
                  return;
                }
                bytecodes_pointer_push(posttest - BOUND_DICTIONARY);

                // Ordinarily, a redirection like the above would count as one.
                count_instructions_++;
              }
              break;
            }

            case CODE_EXIT: {
              I exitdepth = bytecode_get();
              bytecodes_pointer_where()++;
              recursion_current_depth_ -= exitdepth;
              while (do_current_depth_ != 0  &&
                     do_abs_recursion_depth() != recursion_current_depth_) {
                do_current_depth_--;
              }

              count_instructions_++;
              if (single_step) {
                if (is_segment_done()) {
                  bytecodes_pointer_pop();
                }
                return;
              }

              // StackOverflow said I could: https://stackoverflow.com/a/1257776/1623645
              //
              // (I need to 'break' out of a loop, but we're in a switch statement,
              // so 'break' won't apply to the looping structure. I think this is the
              // first 'goto' I've written since I was writing in BASIC (c. 1985).
              goto after_end_of_segment;
            }

            case CODE_PUT: {
              I num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T value = stack_pop();
              variables_[(IndexTypeOf<T>)num] = value;
              break;
            }

            case CODE_INC: {
              I num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T value = stack_pop();
              variables_[(IndexTypeOf<T>)num] += value;
              break;
            }

            case CODE_GET: {
              I num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(variables_[(IndexTypeOf<T>)num]);
              break;
            }

            case CODE_ENUM: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              I start = bytecode_get();
              bytecodes_pointer_where()++;
              I stop = bytecode_get();
              bytecodes_pointer_where()++;
              T result = (T)(current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->read_enum(strings_, start, stop));
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(result);
              break;
            }

            case CODE_ENUMONLY: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              I start = bytecode_get();
              bytecodes_pointer_where()++;
              I stop = bytecode_get();
              bytecodes_pointer_where()++;
              T result = (T)(current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->read_enum(strings_, start, stop));
              if (result == -1) {
                current_error_ = util::ForthError::enumeration_missing;
                return;
              }
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(result);
              break;
            }

            case CODE_PEEK: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T after = stack_pop();
              T result = current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->peek_byte(
                  after, current_error_
              );
              if (current_error_ != util::ForthError::none) {
                return;
              }
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(result);
              break;
            }

            case CODE_LEN_INPUT: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->len());
              break;
            }

            case CODE_POS: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->pos());
              break;
            }

            case CODE_END: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->end() ? -1 : 0);
              break;
            }

            case CODE_SEEK: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->seek(stack_pop(), current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }
              break;
            }

            case CODE_SKIP: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->skip(stack_pop(), current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }
              break;
            }

            case CODE_SKIPWS: {
              I in_num = bytecode_get();
              bytecodes_pointer_where()++;
              current_inputs_[(IndexTypeOf<int64_t>)in_num].get()->skipws();
              break;
            }

            case CODE_WRITE: {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              write_from_stack(out_num, top);
              stack_depth_--;

              count_writes_++;
              break;
            }

            case CODE_WRITE_ADD: {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              write_add_from_stack(out_num, top);
              stack_depth_--;

              count_writes_++;
              break;
            }

            case CODE_WRITE_DUP: {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              current_outputs_[(IndexTypeOf<int64_t>)out_num].get()->dup(stack_pop(), current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }

              count_writes_++;
              break;
            }

            case CODE_LEN_OUTPUT: {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)current_outputs_[(IndexTypeOf<int64_t>)out_num].get()->len());
              break;
            }

            case CODE_REWIND: {
              I out_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              current_outputs_[(IndexTypeOf<int64_t>)out_num].get()->rewind(stack_pop(), current_error_);
              if (current_error_ != util::ForthError::none) {
                return;
              }
              break;
            }

            case CODE_STRING: {
              I string_num = bytecode_get();
              bytecodes_pointer_where()++;
              if (stack_depth_ + 1 >= stack_max_depth_) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)string_num);
              stack_push((T)strings_[(size_t)string_num].size());
              break;
            }

            case CODE_PRINT_STRING: {
              I string_num = bytecode_get();
              bytecodes_pointer_where()++;
              printf("%s", strings_[(IndexTypeOf<int64_t>)string_num].c_str());
              break;
            }

            case CODE_PRINT: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              print_number(stack_pop());
              break;
            }

            case CODE_PRINT_CR: {
              printf("\n");
              break;
            }

            case CODE_PRINT_STACK: {
              printf("<%lld> ", (long long int)stack_depth_);
              for (int64_t i = 0;  i < stack_depth_;  i++) {
                print_number(stack_buffer_[i]);
              }
              printf("<- top ");
              break;
            }

            case CODE_I: {
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)do_i());
              break;
            }

            case CODE_J: {
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)do_j());
              break;
            }

            case CODE_K: {
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)do_k());
              break;
            }

            case CODE_DUP: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_buffer_[stack_depth_] = stack_buffer_[stack_depth_ - 1];
              stack_depth_++;
              break;
            }

            case CODE_DROP: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              stack_depth_--;
              break;
            }

            case CODE_SWAP: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              int64_t tmp = stack_buffer_[stack_depth_ - 2];
              stack_buffer_[stack_depth_ - 2] = stack_buffer_[stack_depth_ - 1];
              stack_buffer_[stack_depth_ - 1] = (T)tmp;
              break;
            }

            case CODE_OVER: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(stack_buffer_[stack_depth_ - 2]);
              break;
            }

            case CODE_ROT: {
              if (stack_cannot_pop3()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              int64_t tmp1 = stack_buffer_[stack_depth_ - 3];
              stack_buffer_[stack_depth_ - 3] = stack_buffer_[stack_depth_ - 2];
              stack_buffer_[stack_depth_ - 2] = stack_buffer_[stack_depth_ - 1];
              stack_buffer_[stack_depth_ - 1] = (T)tmp1;
              break;
            }

            case CODE_NIP: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              stack_buffer_[stack_depth_ - 2] = stack_buffer_[stack_depth_ - 1];
              stack_depth_--;
              break;
            }

            case CODE_TUCK: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              int64_t tmp = stack_buffer_[stack_depth_ - 1];
              stack_buffer_[stack_depth_ - 1] = stack_buffer_[stack_depth_ - 2];
              stack_buffer_[stack_depth_ - 2] = (T)tmp;
              stack_push((T)tmp);
              break;
            }

            case CODE_ADD: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] + pair[1];
              break;
            }

            case CODE_SUB: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] - pair[1];
              break;
            }

            case CODE_MUL: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] * pair[1];
              break;
            }

            case CODE_DIV: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              if (pair[1] == 0) {
                current_error_ = util::ForthError::division_by_zero;
                return;
              }
              // Forth (gforth, at least) does floor division; C++ does integer division.
              // This makes a difference for negative numerator or denominator.
              T tmp = pair[0] / pair[1];
              pair[0] = tmp * pair[1] == pair[0] ? tmp : tmp - ((pair[0] < 0) ^ (pair[1] < 0));
              break;
            }

            case CODE_MOD: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              if (pair[1] == 0) {
                current_error_ = util::ForthError::division_by_zero;
                return;
              }
              // Forth (gforth, at least) does modulo; C++ does remainder.
              // This makes a difference for negative numerator or denominator.
              pair[0] = (pair[1] + (pair[0] % pair[1])) % pair[1];
              break;
            }

            case CODE_DIVMOD: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T one = stack_buffer_[stack_depth_ - 2];
              T two = stack_buffer_[stack_depth_ - 1];
              if (two == 0) {
                current_error_ = util::ForthError::division_by_zero;
                return;
              }
              // See notes on division and modulo/remainder above.
              T tmp = one / two;
              stack_buffer_[stack_depth_ - 1] =
                  tmp * two == one ? tmp : tmp - ((one < 0) ^ (two < 0));
              stack_buffer_[stack_depth_ - 2] =
                  (two + (one % two)) % two;
              break;
            }

            case CODE_NEGATE: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              *top = -(*top);
              break;
            }

            case CODE_ADD1: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              (*top)++;
              break;
            }

            case CODE_SUB1: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              (*top)--;
              break;
            }

            case CODE_ABS: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              *top = abs(*top);
              break;
            }

            case CODE_MIN: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = std::min(pair[0], pair[1]);
              break;
            }

            case CODE_MAX: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = std::max(pair[0], pair[1]);
              break;
            }

            case CODE_EQ: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] == pair[1] ? -1 : 0;
              break;
            }

            case CODE_NE: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] != pair[1] ? -1 : 0;
              break;
            }

            case CODE_GT: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] > pair[1] ? -1 : 0;
              break;
            }

            case CODE_GE: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] >= pair[1] ? -1 : 0;
              break;
            }

            case CODE_LT: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] < pair[1] ? -1 : 0;
              break;
            }

            case CODE_LE: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] <= pair[1] ? -1 : 0;
              break;
            }

            case CODE_EQ0: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              *top = *top == 0 ? -1 : 0;
              break;
            }

            case CODE_INVERT: {
              if (stack_cannot_pop()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* top = stack_peek();
              *top = ~(*top);
              break;
            }

            case CODE_AND: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] & pair[1];
              break;
            }

            case CODE_OR: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] | pair[1];
              break;
            }

            case CODE_XOR: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] ^ pair[1];
              break;
            }

            case CODE_LSHIFT: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] << pair[1];
              break;
            }

            case CODE_RSHIFT: {
              if (stack_cannot_pop2()) {
                current_error_ = util::ForthError::stack_underflow;
                return;
              }
              T* pair = stack_pop2_before_pushing1();
              pair[0] = pair[0] >> pair[1];
              break;
            }

            case CODE_FALSE: {
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(0);
              break;
            }

            case CODE_TRUE: {
              if (stack_cannot_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push(-1);
              break;
            }
          }
        } // end handle one instruction

        count_instructions_++;
        if (single_step) {
          if (is_segment_done()) {
            bytecodes_pointer_pop();
          }
          return;
        }

      } // end walk over instructions in this segment

    after_end_of_segment:
      bytecodes_pointer_pop();

      if (do_current_depth_ != 0  &&
          do_abs_recursion_depth() == recursion_current_depth_) {
        // End one step of a 'do ... loop' or a 'do ... +loop'.
        if (do_loop_is_step()) {
          if (stack_cannot_pop()) {
            current_error_ = util::ForthError::stack_underflow;
            return;
          }
          do_i() += stack_pop();
        }
        else {
          do_i()++;
        }
      }

    } // end of all segments
  }

  template <>
  void
  ForthMachineOf<int32_t, int32_t>::write_from_stack(int64_t num, int32_t* top) noexcept {
    current_outputs_[(IndexTypeOf<int64_t>)num].get()->write_one_int32(*top, false);
  }

  template <>
  void
  ForthMachineOf<int64_t, int32_t>::write_from_stack(int64_t num, int64_t* top) noexcept {
    current_outputs_[(IndexTypeOf<int64_t>)num].get()->write_one_int64(*top, false);
  }

  template <>
  void
  ForthMachineOf<int32_t, int32_t>::write_add_from_stack(int64_t num, int32_t* top) noexcept {
    current_outputs_[(IndexTypeOf<int64_t>)num].get()->write_add_int32(*top);
  }

  template <>
  void
  ForthMachineOf<int64_t, int32_t>::write_add_from_stack(int64_t num, int64_t* top) noexcept {
    current_outputs_[(IndexTypeOf<int64_t>)num].get()->write_add_int64(*top);
  }

  template <>
  void
  ForthMachineOf<int32_t, int32_t>::print_number(int32_t num) noexcept {
    printf("%d ", num);
  }

  template <>
  void
  ForthMachineOf<int64_t, int32_t>::print_number(int64_t num) noexcept {
    printf("%lld ", (long long int)num);
  }

  template class EXPORT_TEMPLATE_INST ForthMachineOf<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST ForthMachineOf<int64_t, int32_t>;

}

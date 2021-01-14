// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthMachine.cpp", line)

#include <sstream>
#include <stdexcept>

#include "awkward/forth/ForthMachine.h"

namespace awkward {
  // Instruction values are preprocessor macros to be equally usable in 32-bit and
  // 64-bit instruction sets.

  // parser flags (parsers are combined bitwise and then bit-inverted to be negative)
  #define PARSER_DIRECT 1
  #define PARSER_REPEATED 2
  #define PARSER_BIGENDIAN 4
  // parser sequential values (starting in the fourth bit)
  #define PARSER_MASK (~(-0x80) & (-0x8))
  #define PARSER_BOOL (0x8 * 1)
  #define PARSER_INT8 (0x8 * 2)
  #define PARSER_INT16 (0x8 * 3)
  #define PARSER_INT32 (0x8 * 4)
  #define PARSER_INT64 (0x8 * 5)
  #define PARSER_INTP (0x8 * 6)
  #define PARSER_UINT8 (0x8 * 7)
  #define PARSER_UINT16 (0x8 * 8)
  #define PARSER_UINT32 (0x8 * 9)
  #define PARSER_UINT64 (0x8 * 10)
  #define PARSER_UINTP (0x8 * 11)
  #define PARSER_FLOAT32 (0x8 * 12)
  #define PARSER_FLOAT64 (0x8 * 13)

  // instructions from special parsing rules
  #define INSTR_LITERAL 0
  #define INSTR_HALT 1
  #define INSTR_PAUSE 2
  #define INSTR_IF 3
  #define INSTR_IF_ELSE 4
  #define INSTR_DO 5
  #define INSTR_DO_STEP 6
  #define INSTR_AGAIN 7
  #define INSTR_UNTIL 8
  #define INSTR_WHILE 9
  #define INSTR_EXIT 10
  #define INSTR_PUT 11
  #define INSTR_INC 12
  #define INSTR_GET 13
  #define INSTR_LEN_INPUT 14
  #define INSTR_POS 15
  #define INSTR_END 16
  #define INSTR_SEEK 17
  #define INSTR_SKIP 18
  #define INSTR_WRITE 19
  #define INSTR_LEN_OUTPUT 20
  #define INSTR_REWIND 21
  // generic builtin instructions
  #define INSTR_I 22
  #define INSTR_J 23
  #define INSTR_K 24
  #define INSTR_DUP 25
  #define INSTR_DROP 26
  #define INSTR_SWAP 27
  #define INSTR_OVER 28
  #define INSTR_ROT 29
  #define INSTR_NIP 30
  #define INSTR_TUCK 31
  #define INSTR_ADD 32
  #define INSTR_SUB 33
  #define INSTR_MUL 34
  #define INSTR_DIV 35
  #define INSTR_MOD 36
  #define INSTR_DIVMOD 37
  #define INSTR_NEGATE 38
  #define INSTR_ADD1 39
  #define INSTR_SUB1 40
  #define INSTR_ABS 41
  #define INSTR_MIN 42
  #define INSTR_MAX 43
  #define INSTR_EQ 44
  #define INSTR_NE 45
  #define INSTR_GT 46
  #define INSTR_GE 47
  #define INSTR_LT 48
  #define INSTR_LE 49
  #define INSTR_EQ0 50
  #define INSTR_INVERT 51
  #define INSTR_AND 52
  #define INSTR_OR 53
  #define INSTR_XOR 54
  #define INSTR_LSHIFT 55
  #define INSTR_RSHIFT 56
  #define INSTR_FALSE 57
  #define INSTR_TRUE 58
  // beginning of the user-defined dictionary
  #define BOUND_DICTIONARY 59

  const std::set<std::string> reserved_words_({
    // comments
    "(", ")", "\\", "\n", "",
    // defining functinos
    ":", ";", "recurse",
    // declaring globals
    "variable", "input", "output",
    // manipulate control flow externally
    "halt", "pause",
    // conditionals
    "if", "then", "else",
    // loops
    "do", "loop", "+loop",
    "begin", "again", "until", "while", "repeat",
    // nonlocal exits
    "exit",
    // variable access
    "!", "+!", "@",
    // input actions
    "len", "pos", "end", "seek", "skip",
    // output actions
    "<-", "stack", "rewind"
  });

  const std::set<std::string> input_parser_words_({
    // single little-endian
    "?->", "b->", "h->", "i->", "q->", "n->", "B->", "H->", "I->", "Q->", "N->", "f->", "d->",
    // single big-endian
    "!h->", "!i->", "!q->", "!n->", "!H->", "!I->", "!Q->", "!N->", "!f->", "!d->",
    // multiple little-endian
    "#?->", "#b->", "#h->", "#i->", "#q->", "#n->", "#B->", "#H->", "#I->", "#Q->", "#N->", "#f->", "#d->",
    // multiple big-endian
    "#!h->", "#!i->", "#!q->", "#!n->", "#!H->", "#!I->", "#!Q->", "#!N->", "#!f->", "#!d->",
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
    // loop variables
    {"i", INSTR_I},
    {"j", INSTR_J},
    {"k", INSTR_K},
    // stack operations
    {"dup", INSTR_DUP},
    {"drop", INSTR_DROP},
    {"swap", INSTR_SWAP},
    {"over", INSTR_OVER},
    {"rot", INSTR_ROT},
    {"nip", INSTR_NIP},
    {"tuck", INSTR_TUCK},
    // basic mathematical functions
    {"+", INSTR_ADD},
    {"-", INSTR_SUB},
    {"*", INSTR_MUL},
    {"/", INSTR_DIV},
    {"mod", INSTR_MOD},
    {"/mod", INSTR_DIVMOD},
    {"negate", INSTR_NEGATE},
    {"1+", INSTR_ADD1},
    {"1-", INSTR_SUB1},
    {"abs", INSTR_ABS},
    {"min", INSTR_MIN},
    {"max", INSTR_MAX},
    // comparisons
    {"=", INSTR_EQ},
    {"<>", INSTR_NE},
    {">", INSTR_GT},
    {">=", INSTR_GE},
    {"<", INSTR_LT},
    {"<=", INSTR_LE},
    {"0=", INSTR_EQ0},
    // bitwise operations
    {"invert", INSTR_INVERT},
    {"and", INSTR_AND},
    {"or", INSTR_OR},
    {"xor", INSTR_XOR},
    {"lshift", INSTR_LSHIFT},
    {"rshift", INSTR_RSHIFT},
    // constants
    {"false", INSTR_FALSE},
    {"true", INSTR_TRUE}
  });

  template <typename T, typename I>
  ForthMachineOf<T, I>::ForthMachineOf(const std::string& source,
                                       int64_t stack_max_depth,
                                       int64_t recursion_max_depth,
                                       int64_t output_initial_size,
                                       double output_resize_factor)
    : source_(source)
    , output_initial_size_(output_initial_size)
    , output_resize_factor_(output_resize_factor)

    , stack_buffer_(new T[stack_max_depth])
    , stack_depth_(0)
    , stack_max_depth_(stack_max_depth)

    , current_inputs_()
    , current_outputs_()
    , ready_(false)

    , current_pause_depth_(0)

    , current_which_(new int64_t[recursion_max_depth])
    , current_where_(new int64_t[recursion_max_depth])
    , recursion_current_depth_(0)
    , recursion_max_depth_(recursion_max_depth)

    , do_recursion_depth_(new int64_t[recursion_max_depth])
    , do_stop_(new int64_t[recursion_max_depth])
    , do_i_(new int64_t[recursion_max_depth])
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
    delete [] current_which_;
    delete [] current_where_;
    delete [] do_recursion_depth_;
    delete [] do_stop_;
    delete [] do_i_;
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::source() const noexcept {
    return source_;
  }

  template <typename T, typename I>
  const ContentPtr
  ForthMachineOf<T, I>::bytecodes() const {
    IndexOf<I> content(bytecodes_.size(), kernel::lib::cpu);
    std::memcpy(content.data(), bytecodes_.data(), bytecodes_.size() * sizeof(I));

    IndexOf<int64_t> offsets(bytecodes_offsets_.size(), kernel::lib::cpu);
    std::memcpy(offsets.data(), bytecodes_offsets_.data(), bytecodes_offsets_.size() * sizeof(int64_t));

    return std::make_shared<ListOffsetArrayOf<int64_t>>(Identities::none(),
                                                        util::Parameters(),
                                                        offsets,
                                                        std::make_shared<NumpyArray>(content),
                                                        false);
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::assembly_instructions() const {
    std::stringstream out;
    for (auto pair : dictionary_names_) {
      int64_t segment_position = bytecodes_offsets_[pair.second - BOUND_DICTIONARY + 1];
      out << ": " << pair.first << std::endl
          << assembly_instruction_segment(segment_position, "  ")
          << ";" << std::endl << std::endl;
    }
    out << assembly_instruction_segment(0);
    return out.str();
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::assembly_instruction_segment(int64_t segment_position,
                                                     const std::string& indent) const {
    if (segment_position < 0  ||  segment_position + 1 >= bytecodes_offsets_.size()) {
      return "";
    }
    std::stringstream out;
    int64_t bytecode_position = bytecodes_offsets_[segment_position];
    int64_t instruction_number = 0;
    while (bytecode_position < bytecodes_offsets_[segment_position + 1]) {
      if (bytecode_position != bytecodes_offsets_[segment_position]) {
        out << indent;
      }
      out << assembly_instruction_at(bytecode_position, indent) << std::endl;
      bytecode_position += bytecodes_per_instruction(bytecode_position);
    }
    return out.str();
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::assembly_instruction_at(int64_t bytecode_position,
                                                const std::string& indent) const {
    if (bytecode_position < 0  ||  bytecode_position >= bytecodes_.size()) {
      return "";
    }

    I bytecode = bytecodes_[bytecode_position];

    if (bytecode < 0) {
      I in_num = bytecodes_[bytecode_position + 1];
      std::string in_name = input_names_[in_num];

      std::string rep = (~bytecode & PARSER_REPEATED) ? "#" : "";
      std::string big = ((~bytecode & PARSER_BIGENDIAN) != 0) ? "!" : "";
      std::string rest;
      switch (~bytecode & PARSER_MASK) {
        case PARSER_BOOL:
          rest = "?->";
          break;
        case PARSER_INT8:
          rest = "b->";
          break;
        case PARSER_INT16:
          rest = "h->";
          break;
        case PARSER_INT32:
          rest = "i->";
          break;
        case PARSER_INT64:
          rest = "q->";
          break;
        case PARSER_INTP:
          rest = "n->";
          break;
        case PARSER_UINT8:
          rest = "B->";
          break;
        case PARSER_UINT16:
          rest = "H->";
          break;
        case PARSER_UINT32:
          rest = "I->";
          break;
        case PARSER_UINT64:
          rest = "Q->";
          break;
        case PARSER_UINTP:
          rest = "N->";
          break;
        case PARSER_FLOAT32:
          rest = "f->";
          break;
        case PARSER_FLOAT64:
          rest = "d->";
          break;
      }
      std::string arrow = rep + big + rest;

      std::string out_name = "stack";
      if (~bytecode & PARSER_REPEATED) {
        I out_num = bytecodes_[bytecode_position + 1];
        out_name = output_names_[out_num];
      }
      return in_name + std::string(" ") + arrow + std::string(" ") + out_name;
    }

    else if (bytecode >= BOUND_DICTIONARY) {
      for (auto pair : dictionary_names_) {
        if (pair.second == bytecode - BOUND_DICTIONARY) {
          return pair.first;
        }
      }
      return "(anonymous segment at " + std::to_string(bytecode - BOUND_DICTIONARY) + ")";
    }

    else {
      switch (bytecode) {
        case INSTR_LITERAL: {
          return std::to_string(bytecodes_[bytecode_position + 1]);
        }
        case INSTR_HALT: {
          return "halt";
        }
        case INSTR_PAUSE: {
          return "pause";
        }
        case INSTR_IF: {
          int64_t consequent = bytecodes_[bytecode_position + 1] - BOUND_DICTIONARY;
          return std::string("if\n")
                 + indent + "  "
                 + assembly_instruction_segment(consequent, indent + "  ")
                 + "then";
        }

        case INSTR_IF_ELSE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_DO: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_DO_STEP: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_AGAIN: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_UNTIL: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_WHILE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_EXIT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_PUT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_INC: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_GET: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_LEN_INPUT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_POS: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_END: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_SEEK: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_SKIP: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_WRITE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_LEN_OUTPUT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_REWIND: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_I: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_J: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_K: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_DUP: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_DROP: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_SWAP: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_OVER: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_ROT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_NIP: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_TUCK: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_ADD: {
          return "+";
        }

        case INSTR_SUB: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_MUL: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_DIV: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_MOD: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_DIVMOD: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_NEGATE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_ADD1: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_SUB1: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_ABS: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_MIN: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_MAX: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_EQ: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_NE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_GT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_GE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_LT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_LE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_EQ0: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_INVERT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_AND: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_OR: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_XOR: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_LSHIFT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_RSHIFT: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_FALSE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }

        case INSTR_TRUE: {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          break;
        }
      }

      return std::string("(unrecognized bytecode ") + std::to_string(bytecode) + ")";
    }
  }

  template <typename T, typename I>
  const std::vector<std::string>
  ForthMachineOf<T, I>::dictionary() const {
    std::vector<std::string> out;
    for (auto pair : dictionary_names_) {
      out.push_back(pair.first);
    }
    return out;
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
    for (int64_t i = 0;  i < variable_names_.size();  i++) {
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
    for (int64_t i = 0;  i < variable_names_.size();  i++) {
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
    return variables_[index];
  }

  template <typename T, typename I>
  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>
  ForthMachineOf<T, I>::outputs() const {
    std::map<std::string, std::shared_ptr<ForthOutputBuffer>> out;
    for (int64_t i = 0;  i < output_names_.size();  i++) {
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
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
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
    return current_outputs_[index];
  }

  template <typename T, typename I>
  const ContentPtr
  ForthMachineOf<T, I>::output_NumpyArray_at(const std::string& name) const {
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i].get()->toNumpyArray();
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const ContentPtr
  ForthMachineOf<T, I>::output_NumpyArray_at(int64_t index) const {
    return current_outputs_[index].get()->toNumpyArray();
  }

  template <typename T, typename I>
  const Index8
  ForthMachineOf<T, I>::output_Index8_at(const std::string& name) const {
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i].get()->toIndex8();
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const Index8
  ForthMachineOf<T, I>::output_Index8_at(int64_t index) const {
    return current_outputs_[index].get()->toIndex8();
  }

  template <typename T, typename I>
  const IndexU8
  ForthMachineOf<T, I>::output_IndexU8_at(const std::string& name) const {
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i].get()->toIndexU8();
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const IndexU8
  ForthMachineOf<T, I>::output_IndexU8_at(int64_t index) const {
    return current_outputs_[index].get()->toIndexU8();
  }

  template <typename T, typename I>
  const Index32
  ForthMachineOf<T, I>::output_Index32_at(const std::string& name) const {
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i].get()->toIndex32();
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const Index32
  ForthMachineOf<T, I>::output_Index32_at(int64_t index) const {
    return current_outputs_[index].get()->toIndex32();
  }

  template <typename T, typename I>
  const IndexU32
  ForthMachineOf<T, I>::output_IndexU32_at(const std::string& name) const {
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i].get()->toIndexU32();
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const IndexU32
  ForthMachineOf<T, I>::output_IndexU32_at(int64_t index) const {
    return current_outputs_[index].get()->toIndexU32();
  }

  template <typename T, typename I>
  const Index64
  ForthMachineOf<T, I>::output_Index64_at(const std::string& name) const {
    if (!is_ready()) {
      throw std::invalid_argument(
        std::string("need to 'begin' or 'run' to create outputs") + FILENAME(__LINE__)
      );
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return current_outputs_[i].get()->toIndex64();
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  template <typename T, typename I>
  const Index64
  ForthMachineOf<T, I>::output_Index64_at(int64_t index) const {
    return current_outputs_[index].get()->toIndex64();
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::reset() {
    stack_depth_ = 0;
    current_inputs_.clear();
    current_outputs_.clear();
    ready_ = false;
    current_pause_depth_ = 0;
    recursion_current_depth_ = 0;
    do_current_depth_ = 0;
    current_error_ = util::ForthError::none;
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::begin(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::begin() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::step() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::maybe_throw(util::ForthError err,
                                    const std::set<util::ForthError>& ignore) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::run() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::resume() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::call(const std::string& name) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::call(int64_t index) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::pause_depth() const noexcept {
    return current_pause_depth_;
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::current_bytecode() const noexcept {
    if (recursion_current_depth_ == 0) {
      return -1;
    }
    else {
      return current_where_[recursion_current_depth_ - 1];
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
        value = std::stoul(word.substr(2, (int64_t)word.size() - 2), nullptr, 16);
      }
      catch (std::invalid_argument err) {
        return false;
      }
      return true;
    }
    else {
      try {
        value = std::stoul(word, nullptr, 10);
      }
      catch (std::invalid_argument err) {
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
  ForthMachineOf<T, I>::is_reserved(const std::string& word) const {
    return reserved_words_.find(word) != reserved_words_.end()  ||
           input_parser_words_.find(word) != input_parser_words_.end()  ||
           output_dtype_words_.find(word) != output_dtype_words_.end()  ||
           generic_builtin_words_.find(word) != generic_builtin_words_.end();
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_defined(const std::string& word) const {
    return dictionary_names_.find(word) != dictionary_names_.end();
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::bytecodes_per_instruction(int64_t bytecode_position) const {
    I bytecode = bytecodes_[bytecode_position];
    I next_bytecode = 0;
    if (bytecode_position + 1 < bytecodes_.size()) {
      next_bytecode = bytecodes_[bytecode_position + 1];
    }

    if (bytecode < 0) {
      if (~bytecode & PARSER_DIRECT) {
        return 3;
      }
      else {
        return 2;
      }
    }
    else if (next_bytecode == INSTR_AGAIN  ||  next_bytecode == INSTR_UNTIL) {
      return 2;
    }
    else if (next_bytecode == INSTR_WHILE) {
      return 3;
    }
    else {
      switch (bytecode) {
        case INSTR_IF_ELSE:
          return 3;
        case INSTR_LITERAL:
        case INSTR_IF:
        case INSTR_DO:
        case INSTR_DO_STEP:
        case INSTR_EXIT:
        case INSTR_PUT:
        case INSTR_INC:
        case INSTR_GET:
        case INSTR_LEN_INPUT:
        case INSTR_POS:
        case INSTR_END:
        case INSTR_SEEK:
        case INSTR_SKIP:
        case INSTR_WRITE:
        case INSTR_LEN_OUTPUT:
        case INSTR_REWIND:
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
    std::pair<int64_t, int64_t> lc = linecol[startpos];
    std::stringstream out;
    out << "in AwkwardForth source code, line " << lc.first << " col " << lc.second
        << ", " << message << ":" << std::endl << std::endl << "    ";
    int64_t line = 1;
    int64_t col = 1;
    int64_t start = 0;
    int64_t stop = 0;
    while (stop < source_.length()) {
      if (lc.first == line  &&  lc.second == col) {
        start = stop;
      }
      if (stoppos < linecol.size()  &&
          linecol[stoppos].first == line  &&  linecol[stoppos].second == col) {
        break;
      }
      if (source_[stop] == '\n') {
        line += 1;
        col = 0;
      }
      col++;
      stop++;
    }
    out << source_.substr(start, stop - start);
    return out.str();
  }

  template <typename T, typename I>
  void ForthMachineOf<T, I>::tokenize(std::vector<std::string>& tokenized,
                                      std::vector<std::pair<int64_t, int64_t>>& linecol) {
    int64_t start = 0;
    int64_t stop = 0;
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
        line += 1;
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
    }
    // The source code might end on non-whitespace.
    if (full) {
      tokenized.push_back(source_.substr(start, stop - start));
      linecol.push_back(std::pair<int64_t, int64_t>(line, colstart));
    }
  }

  template <typename T, typename I>
  void ForthMachineOf<T, I>::compile(const std::vector<std::string>& tokenized,
                                     const std::vector<std::pair<int64_t, int64_t>>& linecol) {
    std::vector<std::vector<I>> dictionary;

    // Start recursive parsing.
    std::vector<I> bytecodes;
    dictionary.push_back(bytecodes);
    parse("",
          tokenized,
          linecol,
          0,
          tokenized.size(),
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
      bytecodes_offsets_.push_back(bytecodes_.size());
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
                              int64_t dodepth) const {
    int64_t pos = start;
    while (pos < stop) {
      std::string word = tokenized[pos];

      if (word == "(") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "\\") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "\n") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == ":") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "recurse") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "variable") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "input") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "output") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
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
            );
          }
          else if (tokenized[substop] == "if") {
            nesting++;
          }
          else if (tokenized[substop] == "then") {
            nesting--;
          }
          else if (tokenized[substop] == "else" and nesting == 1) {
            subelse = substop;
          }
        }

        if (subelse == -1) {
          // Add the consequent to the dictionary so that it can be used
          // without special instruction pointer manipulation at runtime.
          I bytecode = dictionary.size() + BOUND_DICTIONARY;
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
          dictionary[bytecode - BOUND_DICTIONARY] = consequent;

          bytecodes.push_back(INSTR_IF);
          bytecodes.push_back(bytecode);

          pos = substop + 1;
        }
        else {
          // Same as above, except that two new definitions must be made.
          I bytecode1 = dictionary.size() + BOUND_DICTIONARY;
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
          dictionary[bytecode1 - BOUND_DICTIONARY] = consequent;

          I bytecode2 = dictionary.size() + BOUND_DICTIONARY;
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
          dictionary[bytecode2 - BOUND_DICTIONARY] = alternate;

          bytecodes.push_back(INSTR_IF_ELSE);
          bytecodes.push_back(bytecode1);
          bytecodes.push_back(bytecode2);

          pos = substop + 1;
        }
      }

      else if (word == "do") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "begin") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (word == "exit") {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (is_variable(word)) {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (is_input(word)) {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else if (is_output(word)) {
        throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
      }

      else {
        auto generic_builtin = generic_builtin_words_.find(word);
        if (generic_builtin != generic_builtin_words_.end()) {
          if (word == "i"  &&  dodepth < 1) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 1, "only allowed in a 'do' loop")
            );
          }
          if (word == "j"  &&  dodepth < 2) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 1, "only allowed in a nested 'do' loop")
            );
          }
          if (word == "k"  &&  dodepth < 3) {
            throw std::invalid_argument(
              err_linecol(linecol, pos, pos + 1, "only allowed in a doubly nested 'do' loop")
            );
          }
          bytecodes.push_back(generic_builtin->second);

          pos++;
        }

        else {
          auto pair = dictionary_names_.find(word);
          if (pair != dictionary_names_.end()) {
            throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
          }

          else {
            int64_t num;
            if (is_integer(word, num)) {
              bytecodes.push_back(INSTR_LITERAL);
              bytecodes.push_back(num);

              pos++;
            }

            else {
              throw std::invalid_argument(
                err_linecol(linecol, pos, pos + 1,
                            "unrecognized word or wrong context for word")
              );
            }
          } // check is_integer
        } // check is_defined
      } // check is_generic_builtin
    } // end loop over segment
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::internal_run(bool keep_going) { // noexcept
    while (recursion_current_depth_ != 0) {
      while (bytecodes_pointer_where() < (
                 bytecodes_offsets_[bytecodes_pointer_which() + 1] -
                 bytecodes_offsets_[bytecodes_pointer_which()]
             )) {
        I bytecode = bytecode_get();

        if (do_current_depth_ == 0  ||  do_abs_recursion_depth() != recursion_current_depth_) {
          // Normal operation: step forward one bytecode.
          bytecodes_pointer_where() += 1;
        }
        else if (do_i() >= do_stop()) {
          // End a 'do' loop.
          do_current_depth_--;
          bytecodes_pointer_where() += 1;
          continue;
        }
        // else... don't increase bytecode_pointer_where()

        if (bytecode < 0) {
          throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
        }

        else if (bytecode >= BOUND_DICTIONARY) {
          if (recursion_current_depth_ == recursion_max_depth_) {
            current_error_ = util::ForthError::recursion_depth_exceeded;
            return;
          }
          bytecodes_pointer_push((bytecode - BOUND_DICTIONARY) + 1);
        }

        else {
          switch (bytecode) {
            case INSTR_LITERAL: {
              I num = bytecode_get();
              bytecodes_pointer_where()++;
              if (!stack_can_push()) {
                current_error_ = util::ForthError::stack_overflow;
                return;
              }
              stack_push((T)num);
              break;
            }

            case INSTR_HALT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_PAUSE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_IF: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_IF_ELSE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_DO: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_DO_STEP: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_AGAIN: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_UNTIL: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_WHILE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_EXIT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_PUT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_INC: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_GET: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_LEN_INPUT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_POS: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_END: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_SEEK: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_SKIP: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_WRITE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_LEN_OUTPUT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_REWIND: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_I: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_J: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_K: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_DUP: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_DROP: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_SWAP: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_OVER: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_ROT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_NIP: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_TUCK: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_ADD: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_SUB: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_MUL: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_DIV: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_MOD: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_DIVMOD: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_NEGATE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_ADD1: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_SUB1: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_ABS: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_MIN: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_MAX: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_EQ: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_NE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_GT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_GE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_LT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_LE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_EQ0: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_INVERT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_AND: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_OR: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_XOR: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_LSHIFT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_RSHIFT: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_FALSE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }

            case INSTR_TRUE: {
              throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
              break;
            }
          }
        } // end handle one instruction

        count_instructions_++;
        if (!keep_going) {
          if (is_segment_done()) {
            bytecodes_pointer_pop();
          }
          return;
        }

      } // end walk over instructions in this segment

    after_end_of_segment:
      bytecodes_pointer_pop();

      if (do_current_depth_ != 0  &&  do_abs_recursion_depth() == recursion_current_depth_) {
        // End one step of a 'do ... loop' or a 'do ... +loop'.
        if (do_loop_is_step()) {
          if (!stack_can_pop()) {
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
    if (num == 1) {
      current_outputs_[num].get()->write_one_int32(*top, false);
    }
    else {
      current_outputs_[num].get()->write_int32(1, top, false);
    }
  }

  template <>
  void
  ForthMachineOf<int64_t, int32_t>::write_from_stack(int64_t num, int64_t* top) noexcept {
    if (num == 1) {
      current_outputs_[num].get()->write_one_int64(*top, false);
    }
    else {
      current_outputs_[num].get()->write_int64(1, top, false);
    }
  }

  template class EXPORT_TEMPLATE_INST ForthMachineOf<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST ForthMachineOf<int64_t, int32_t>;

}

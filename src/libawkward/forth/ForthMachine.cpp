// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthMachine.cpp", line)

#include "awkward/forth/ForthMachine.h"

namespace awkward {
  // Instruction values are preprocessor macros to be equally usable in 32-bit and
  // 64-bit instruction sets.

  // parser flags (parsers are combined bitwise and then bit-inverted to be negative)
  #define PARSER_DIRECT 1
  #define PARSER_REPEATED 2
  #define PARSER_BIGENDIAN 4
  #define PARSER_MASK ~(1 + 2 + 4)
  // parser sequential values (starting in the fourth bit)
  #define PARSER_BOOL 8
  #define PARSER_INT8 16
  #define PARSER_INT16 24
  #define PARSER_INT32 32
  #define PARSER_INT64 40
  #define PARSER_INTP 48
  #define PARSER_UINT8 56
  #define PARSER_UINT16 64
  #define PARSER_UINT32 72
  #define PARSER_UINT64 80
  #define PARSER_UINTP 88
  #define PARSER_FLOAT32 96
  #define PARSER_FLOAT64 104

  // instructions from special parsing rules
  #define INSTR_LITERAL 0
  #define INSTR_IF 1
  #define INSTR_IF_ELSE 2
  #define INSTR_DO 3
  #define INSTR_DO_STEP 4
  #define INSTR_AGAIN 5
  #define INSTR_UNTIL 6
  #define INSTR_WHILE 7
  #define INSTR_EXIT 8
  #define INSTR_PUT 9
  #define INSTR_INC 10
  #define INSTR_GET 11
  #define INSTR_LEN_INPUT 12
  #define INSTR_POS 13
  #define INSTR_END 14
  #define INSTR_SEEK 15
  #define INSTR_SKIP 16
  #define INSTR_WRITE 17
  #define INSTR_LEN_OUTPUT 18
  #define INSTR_REWIND 19
  // generic builtin instructions
  #define INSTR_INDEX_I 20
  #define INSTR_INDEX_J 21
  #define INSTR_INDEX_K 22
  #define INSTR_DUP 23
  #define INSTR_DROP 24
  #define INSTR_SWAP 25
  #define INSTR_OVER 26
  #define INSTR_ROT 27
  #define INSTR_NIP 28
  #define INSTR_TUCK 29
  #define INSTR_ADD 30
  #define INSTR_SUB 31
  #define INSTR_MUL 32
  #define INSTR_DIV 33
  #define INSTR_MOD 34
  #define INSTR_DIVMOD 35
  #define INSTR_NEGATE 36
  #define INSTR_ADD1 37
  #define INSTR_SUB1 38
  #define INSTR_ABS 39
  #define INSTR_MIN 40
  #define INSTR_MAX 41
  #define INSTR_EQ 42
  #define INSTR_NE 43
  #define INSTR_GT 44
  #define INSTR_GE 45
  #define INSTR_LT 46
  #define INSTR_LE 47
  #define INSTR_EQ0 48
  #define INSTR_INVERT 49
  #define INSTR_AND 50
  #define INSTR_OR 51
  #define INSTR_XOR 52
  #define INSTR_LSHIFT 53
  #define INSTR_RSHIFT 54
  #define INSTR_FALSE 55
  #define INSTR_TRUE 56
  // beginning of the user-defined dictionary
  #define BOUND_DICTIONARY 57

  const std::set<std::string> reserved_words_({
    // comments
    "(", ")", "\\", "\n", "",
    // defining functinos
    ":", ";", "recurse",
    // declaring globals
    "variable", "input", "output",
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
    {"i", INSTR_INDEX_I},
    {"j", INSTR_INDEX_J},
    {"k", INSTR_INDEX_K},
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
                                       int64_t stack_size,
                                       int64_t recursion_depth,
                                       int64_t output_initial_size,
                                       double output_resize_factor)
    : source_(source)
    , output_initial_size_(output_initial_size)
    , output_resize_factor_(output_resize_factor)

    , stack_buffer_(new T[stack_size])
    , stack_top_(0)
    , stack_size_(stack_size)

    , current_inputs_()
    , current_outputs_()

    , current_which_(new int64_t[recursion_depth])
    , current_where_(new int64_t[recursion_depth])
    , instruction_current_depth_(0)
    , instruction_max_depth_(recursion_depth)

    , do_instruction_depth_(new int64_t[recursion_depth])
    , do_stop_(new int64_t[recursion_depth])
    , do_i_(new int64_t[recursion_depth])
    , do_current_depth_(0)

    , current_error_(util::ForthError::none)

    , count_instructions_(0)
    , count_reads_(0)
    , count_writes_(0)
    , count_nanoseconds_(0)
  {
    compile();
  }

  template <typename T, typename I>
  ForthMachineOf<T, I>::~ForthMachineOf() {
    delete [] stack_buffer_;
    delete [] current_which_;
    delete [] current_where_;
    delete [] do_instruction_depth_;
    delete [] do_stop_;
    delete [] do_i_;
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::source() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::vector<I>
  ForthMachineOf<T, I>::bytecodes() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::assembly_instructions() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::stack_size() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::recursion_depth() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::output_initial_size() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::output_resize_factor() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::vector<T>
  ForthMachineOf<T, I>::stack() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::map<std::string, T>
  ForthMachineOf<T, I>::variables() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
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
  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>
  ForthMachineOf<T, I>::step() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::current_instruction() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::end() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs,
      const std::set<util::ForthError>& ignore) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::run() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::count_reset() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_instructions() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_reads() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_writes() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::count_nanoseconds() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void ForthMachineOf<T, I>::compile() {

  }

  template class EXPORT_TEMPLATE_INST ForthMachineOf<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST ForthMachineOf<int64_t, int32_t>;

}

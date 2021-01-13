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
  #define INSTR_BREAKPOINT 1
  #define INSTR_IF 2
  #define INSTR_IF_ELSE 3
  #define INSTR_DO 4
  #define INSTR_DO_STEP 5
  #define INSTR_AGAIN 6
  #define INSTR_UNTIL 7
  #define INSTR_WHILE 8
  #define INSTR_EXIT 9
  #define INSTR_PUT 10
  #define INSTR_INC 11
  #define INSTR_GET 12
  #define INSTR_LEN_INPUT 13
  #define INSTR_POS 14
  #define INSTR_END 15
  #define INSTR_SEEK 16
  #define INSTR_SKIP 17
  #define INSTR_WRITE 18
  #define INSTR_LEN_OUTPUT 19
  #define INSTR_REWIND 20
  // generic builtin instructions
  #define INSTR_I 21
  #define INSTR_J 22
  #define INSTR_K 23
  #define INSTR_DUP 24
  #define INSTR_DROP 25
  #define INSTR_SWAP 26
  #define INSTR_OVER 27
  #define INSTR_ROT 28
  #define INSTR_NIP 29
  #define INSTR_TUCK 30
  #define INSTR_ADD 31
  #define INSTR_SUB 32
  #define INSTR_MUL 33
  #define INSTR_DIV 34
  #define INSTR_MOD 35
  #define INSTR_DIVMOD 36
  #define INSTR_NEGATE 37
  #define INSTR_ADD1 38
  #define INSTR_SUB1 39
  #define INSTR_ABS 40
  #define INSTR_MIN 41
  #define INSTR_MAX 42
  #define INSTR_EQ 43
  #define INSTR_NE 44
  #define INSTR_GT 45
  #define INSTR_GE 46
  #define INSTR_LT 47
  #define INSTR_LE 48
  #define INSTR_EQ0 49
  #define INSTR_INVERT 50
  #define INSTR_AND 51
  #define INSTR_OR 52
  #define INSTR_XOR 53
  #define INSTR_LSHIFT 54
  #define INSTR_RSHIFT 55
  #define INSTR_FALSE 56
  #define INSTR_TRUE 57
  // beginning of the user-defined dictionary
  #define BOUND_DICTIONARY 58

  const std::set<std::string> reserved_words_({
    // comments
    "(", ")", "\\", "\n", "",
    // defining functinos
    ":", ";", "recurse",
    // declaring globals
    "variable", "input", "output",
    // resumable control flow
    "breakpoint",
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
    , stack_top_(0)
    , stack_max_depth_(stack_max_depth)

    , current_inputs_()
    , current_outputs_()

    , current_breakpoint_depth_(0)

    , current_which_(new int64_t[recursion_max_depth])
    , current_where_(new int64_t[recursion_max_depth])
    , instruction_current_depth_(0)
    , instruction_max_depth_(recursion_max_depth)

    , do_instruction_depth_(new int64_t[recursion_max_depth])
    , do_stop_(new int64_t[recursion_max_depth])
    , do_i_(new int64_t[recursion_max_depth])
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
  const std::vector<std::string>
  ForthMachineOf<T, I>::dictionary() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::stack_max_depth() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::recursion_max_depth() const {
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
  T
  ForthMachineOf<T, I>::stack_at(int64_t from_top) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::stack_depth() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::stack_can_push() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::stack_can_pop() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::stack_push(T value) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  T
  ForthMachineOf<T, I>::stack_pop() { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::stack_clear() { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::map<std::string, T>
  ForthMachineOf<T, I>::variables() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::vector<std::string>
  ForthMachineOf<T, I>::variable_index() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  T
  ForthMachineOf<T, I>::variable_at(const std::string& name) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  T
  ForthMachineOf<T, I>::variable_at(int64_t index) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>
  ForthMachineOf<T, I>::outputs() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::vector<std::string>
  ForthMachineOf<T, I>::output_index() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::shared_ptr<ForthOutputBuffer>
  ForthMachineOf<T, I>::output_at(const std::string& name) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::shared_ptr<ForthOutputBuffer>
  ForthMachineOf<T, I>::output_at(int64_t index) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::outputs_clear() {
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
  util::ForthError
  ForthMachineOf<T, I>::step() {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  util::ForthError
  ForthMachineOf<T, I>::run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs,
      const std::set<util::ForthError>& ignore) {
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
  ForthMachineOf<T, I>::breakpoint_depth() const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::current_instruction() const {
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
  bool
  ForthMachineOf<T, I>::is_integer(const std::string& word, int64_t& value) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_variable(const std::string& word) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_input(const std::string& word) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_output(const std::string& word) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_reserved(const std::string& word) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_defined(const std::string& word) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void ForthMachineOf<T, I>::compile() {
    // not implemented
  }

  template <typename T, typename I>
  const std::string
  ForthMachineOf<T, I>::err_linecol(const std::vector<std::pair<int64_t, int64_t>>& linecol,
                                    int64_t startpos,
                                    int64_t stoppos,
                                    const std::string& message) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::parse(const std::string& defn,
                              const std::vector<std::string>& tokenized,
                              const std::vector<std::pair<int64_t, int64_t>>& linecol,
                              int64_t start,
                              int64_t stop,
                              std::vector<I>& bytecodes,
                              int64_t exitdepth,
                              int64_t dodepth) const {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::write_from_stack(int64_t num, T* top) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_done() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::is_segment_done() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::internal_run(bool keep_going) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  T*
  ForthMachineOf<T, I>::stack_pop2() { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  T*
  ForthMachineOf<T, I>::stack_pop2_before_pushing1() { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  T*
  ForthMachineOf<T, I>::stack_peek() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  I
  ForthMachineOf<T, I>::instruction_get() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::instruction_pointer_push(int64_t which) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::instruction_pointer_pop() { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::instruction_pointer_which() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::instruction_pointer_where() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::do_loop_push(int64_t start, int64_t stop) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  ForthMachineOf<T, I>::do_steploop_push(int64_t start, int64_t stop) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::do_instruction_depth() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t
  ForthMachineOf<T, I>::do_abs_instruction_depth() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  ForthMachineOf<T, I>::do_loop_is_step() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::do_stop() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::do_i() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::do_j() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  int64_t&
  ForthMachineOf<T, I>::do_k() const { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template class EXPORT_TEMPLATE_INST ForthMachineOf<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST ForthMachineOf<int64_t, int32_t>;

}

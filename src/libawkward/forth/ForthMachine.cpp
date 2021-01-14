// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthMachine.cpp", line)

#include <sstream>

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
    compile();
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
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
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
    if (output_names_.size() != current_outputs_.size()) {
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
    if (output_names_.size() != current_outputs_.size()) {
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
    if (output_names_.size() != current_outputs_.size()) {
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
    if (output_names_.size() != current_outputs_.size()) {
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
    if (output_names_.size() != current_outputs_.size()) {
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
    if (output_names_.size() != current_outputs_.size()) {
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
    if (output_names_.size() != current_outputs_.size()) {
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
  int64_t
  ForthMachineOf<T, I>::current_instruction() const noexcept {
    int64_t bytecode_pos = current_bytecode();
    if (bytecode_pos == -1) {
      return bytecode_pos;
    }
    else {
      return bytecode_to_instruction_[bytecode_pos];
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
  const std::string
  ForthMachineOf<T, I>::err_linecol(const std::vector<std::pair<int64_t, int64_t>>& linecol,
                                    int64_t startpos,
                                    int64_t stoppos,
                                    const std::string& message) const {
    std::pair<int64_t, int64_t> lc = linecol[startpos];
    std::stringstream out;
    out << "in Awkward Forth source code, line " << lc.first << " col " << lc.second
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
  void ForthMachineOf<T, I>::compile() {
    // not implemented

    bytecodes_offsets_.push_back(0);
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
  ForthMachineOf<T, I>::internal_run(bool keep_going) { // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
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

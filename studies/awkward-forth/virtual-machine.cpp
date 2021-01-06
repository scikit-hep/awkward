// c++ virtual-machine.cpp -o virtual-machine-test  &&  ./virtual-machine-test

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <cstring>

#include <iostream>


template <typename T>
class array_deleter {
public:
    void operator()(T const *ptr) {
      delete [] ptr;
    }
};


enum class dtype {
    NOT_PRIMITIVE,
    boolean,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    float128,
    complex64,
    complex128,
    complex256,
    // datetime64,
    // timedelta64,
    size
};


enum class ForthError {
  none,
  stack_underflow,
  read_beyond,
  seek_beyond,
  skip_beyond,
  rewind_beyond,
  size
};


class ForthInputBuffer {
public:
  ForthInputBuffer(const std::shared_ptr<void> ptr,
                   int64_t offset,
                   int64_t length)
    : ptr_(ptr)
    , offset_(offset)
    , length_(length)
    , pos_(0) { }

  void* read(int64_t num_bytes, ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next > length_) {
      err = ForthError::read_beyond;
      return nullptr;
    }
    void* out = reinterpret_cast<void*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)pos_
    );
    pos_ = next;
    return out;
  }

  void seek(int64_t to, ForthError& err) noexcept {
    if (to < 0  ||  to > length_) {
      err = ForthError::seek_beyond;
    }
    else {
      pos_ = to;
    }
  }

  void skip(int64_t num_bytes, ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next < 0  ||  next > length_) {
      err = ForthError::skip_beyond;
    }
    else {
      pos_ = next;
    }
  }

  bool end() const noexcept {
    return pos_ == length_;
  }

  int64_t pos() const noexcept {
    return pos_;
  }

  int64_t len() const noexcept {
    return length_;
  }

private:
  std::shared_ptr<void> ptr_;
  int64_t offset_;
  int64_t length_;
  int64_t pos_;
};


class ForthOutputBuffer {
public:
  ForthOutputBuffer(int64_t initial=1024, double resize=1.5)
    : length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  int64_t length() const noexcept {
    return length_;
  }

  void rewind(int64_t num_items, ForthError& err) noexcept {
    int64_t next = length_ - num_items;
    if (next < 0) {
      err = ForthError::rewind_beyond;
    }
    else {
      length_ = next;
    }
  }

  virtual std::shared_ptr<void> ptr() const noexcept = 0;

  virtual void write_bool(int64_t num_items, const bool* values) noexcept = 0;
  virtual void write_int8(int64_t num_items, const int8_t* values) noexcept = 0;
  virtual void write_int16(int64_t num_items, const int16_t* values) noexcept = 0;
  virtual void write_int32(int64_t num_items, const int32_t* values) noexcept = 0;
  virtual void write_int64(int64_t num_items, const int64_t* values) noexcept = 0;
  virtual void write_intp(int64_t num_items, const ssize_t* values) noexcept = 0;
  virtual void write_uint8(int64_t num_items, const uint8_t* values) noexcept = 0;
  virtual void write_uint16(int64_t num_items, const uint16_t* values) noexcept = 0;
  virtual void write_uint32(int64_t num_items, const uint32_t* values) noexcept = 0;
  virtual void write_uint64(int64_t num_items, const uint64_t* values) noexcept = 0;
  virtual void write_uintp(int64_t num_items, const size_t* values) noexcept = 0;
  virtual void write_float32(int64_t num_items, const float* values) noexcept = 0;
  virtual void write_float64(int64_t num_items, const double* values) noexcept = 0;

protected:
  int64_t length_;
  int64_t reserved_;
  double resize_;
};


template <typename OUT>
class ForthOutputBufferOf : public ForthOutputBuffer {
public:
  ForthOutputBufferOf(int64_t initial=1024, double resize=1.5)
    : ForthOutputBuffer(initial, resize)
    , ptr_(new OUT[initial], array_deleter<OUT>()) { }

  std::shared_ptr<void> ptr() const noexcept override {
    return ptr_;
  }

  void write_bool(int64_t num_items, const bool* values) noexcept override {
    return write(num_items, values);
  }

  void write_int8(int64_t num_items, const int8_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_int16(int64_t num_items, const int16_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_int32(int64_t num_items, const int32_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_int64(int64_t num_items, const int64_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_intp(int64_t num_items, const ssize_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_uint8(int64_t num_items, const uint8_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_uint16(int64_t num_items, const uint16_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_uint32(int64_t num_items, const uint32_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_uint64(int64_t num_items, const uint64_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_uintp(int64_t num_items, const size_t* values) noexcept override {
    return write(num_items, values);
  }

  void write_float32(int64_t num_items, const float* values) noexcept override {
    return write(num_items, values);
  }

  void write_float64(int64_t num_items, const double* values) noexcept override {
    return write(num_items, values);
  }

private:
  template <typename IN>
  void write(int64_t num_items, const IN* values) noexcept {
    int64_t next = length_ + num_items;
    if (next > reserved_) {
      int64_t reservation = reserved_;
      while (next > reservation) {
        reservation = (int64_t)std::ceil(reservation * resize_);
      }
      std::shared_ptr<OUT> new_buffer = std::shared_ptr<OUT>(new OUT[reservation],
                                                             array_deleter<OUT>());
      std::memcpy(new_buffer.get(), ptr_.get(), sizeof(OUT) * reservation);
      ptr_ = new_buffer;
    }
    for (int64_t i = 0;  i < num_items;  i++) {
      ptr_.get()[length_ + i] = values[i];
    }
    length_ = next;
  }

  std::shared_ptr<OUT> ptr_;
};


template <typename T>
class ForthStack {
public:
  ForthStack(int64_t initial=1024, double resize=1.5)
    : buffer_(new T[initial])
    , length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  ~ForthStack() {
    delete [] buffer_;
  }

  int64_t length() const noexcept {
    return length_;
  }

  void push(T value) noexcept {
    if (length_ == reserved_) {
      int64_t reservation = (int64_t)std::ceil(reserved_ * resize_);
      T* new_buffer = new T[reservation];
      std::memcpy(new_buffer, buffer_, sizeof(T) * reserved_);
      delete [] buffer_;
      buffer_ = new_buffer;
      reserved_ = reservation;
    }
    buffer_[length_] = value;
    length_++;
  }

  T pop(ForthError &err) noexcept {
    if (length_ == 0) {
      err = ForthError::stack_underflow;
      return 0;
    }
    else {
      length_--;
      return buffer_[length_];
    }
  }

  void clear() noexcept {
    length_ = 0;
  }

  const std::string tostring() const noexcept {
    std::stringstream out;
    for (int64_t i = 0;  i < length_;  i++) {
      out << buffer_[i] << " ";
    }
    out << "<- top";
    return out.str();
  }

private:
  T* buffer_;
  int64_t length_;
  int64_t reserved_;
  double resize_;
};


class ForthInstructionPointer {
public:
  ForthInstructionPointer(int64_t reservation=1024)
    : which_(new int64_t[reservation])
    , where_(new int64_t[reservation])
    , skip_(new int64_t[reservation])
    , length_(0)
    , reserved_(reservation) { }

  ~ForthInstructionPointer() {
    delete [] which_;
    delete [] where_;
    delete [] skip_;
  }

  bool empty() const noexcept {
    return length_ == 0;
  }

  bool push(int64_t which, int64_t where, int64_t skip) noexcept {
    if (length_ == reserved_) {
      return false;
    }
    which_[length_] = which;
    where_[length_] = where;
    skip_[length_] = skip;
    length_++;
    return true;
  }

  void pop() noexcept {
    length_--;
  }

  int64_t& which() noexcept {
    return which_[length_ - 1];
  }

  int64_t& where() noexcept {
    return where_[length_ - 1];
  }

  int64_t& skip() noexcept {
    return skip_[length_ - 1];
  }

private:
  int64_t* which_;
  int64_t* where_;
  int64_t* skip_;
  int64_t length_;
  int64_t reserved_;
};


#define PARSER_DIRECT 1
#define PARSER_REPEATED 2
#define PARSER_BIGENDIAN 4
#define PARSER_MASK ~(1 + 2 + 4)

#define PARSER_BOOL 8
#define PARSER_INT8 16
#define PARSER_INT16 24
#define PARSER_INT32 32
#define PARSER_INT64 40
#define PARSER_SSIZE 48
#define PARSER_UINT8 56
#define PARSER_UINT16 64
#define PARSER_UINT32 72
#define PARSER_UINT64 80
#define PARSER_USIZE 88
#define PARSER_FLOAT32 96
#define PARSER_FLOAT64 104

#define LITERAL 0
#define PUT 1
#define INC 2
#define GET 3
#define SKIP 4
#define SEEK 5
#define END 6
#define POSITION 7
#define LENGTH_INPUT 8
#define REWIND 9
#define LENGTH_OUTPUT 10
#define WRITE 11
#define IF 12
#define IF_ELSE 13
#define DO 14
#define DO_STEP 15
#define AGAIN 16
#define UNTIL 17
#define WHILE 18
#define EXIT 19
#define INDEX_I 20
#define INDEX_J 21
#define INDEX_K 22
#define DUP 23
#define DROP 24
#define SWAP 25
#define OVER 26
#define ROT 27
#define NIP 28
#define TUCK 29
#define ADD 30
#define SUB 31
#define MUL 32
#define DIV 33
#define MOD 34
#define DIVMOD 35
#define LSHIFT 36
#define RSHIFT 37
#define ABS 38
#define MIN 39
#define MAX 40
#define NEGATE 41
#define ADD1 42
#define SUB1 43
#define EQ0 44
#define EQ 45
#define NE 46
#define GT 47
#define GE 48
#define LT 49
#define LE 50
#define AND 51
#define OR 52
#define XOR 53
#define INVERT 54
#define FALSE 55
#define TRUE 56


template <typename T, typename I, bool DEBUG>
class ForthMachine {
public:
  ForthMachine(const std::string& source,
               int64_t initial_buffer=1024,
               double resize_buffer=1.5,
               int64_t initial_stack=1024,
               double resize_stack=1.5,
               int64_t recursion_depth=1024)
    : initial_buffer_(initial_buffer)
    , resize_buffer_(resize_buffer)
    , stack_(initial_stack, resize_stack)
    , recursion_depth_(recursion_depth) {
    compile(source);
  }

  const ForthStack<T>& stack() const {
    return stack_;
  }

  T variable(const std::string& name) const {
    for (size_t i = 0;  i < variable_names_.size();  i++) {
      if (variable_names_[i] == name) {
        return variables_[i];
      }
    }
    throw std::invalid_argument(
      std::string("unrecognized variable name: ") + name
    );
  }

  std::map<std::string, std::shared_ptr<ForthOutputBuffer>> run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {

    std::vector<std::shared_ptr<ForthInputBuffer>> ins;
    for (auto name : input_names_) {
      auto it = inputs.find(name);
      if (it == inputs.end()) {
        throw std::invalid_argument(
          std::string("name missing from inputs: ") + name
        );
      }
      ins.push_back(it->second);
    }

    std::map<std::string, std::shared_ptr<ForthOutputBuffer>> outputs;
    std::vector<std::shared_ptr<ForthOutputBuffer>> outs;
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      std::string name = output_names_[i];
      std::shared_ptr<ForthOutputBuffer> out;
      switch (output_dtypes_[i]) {
        case dtype::boolean: {
          out = std::make_shared<ForthOutputBufferOf<bool>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int8: {
          out = std::make_shared<ForthOutputBufferOf<int8_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int16: {
          out = std::make_shared<ForthOutputBufferOf<int16_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int32: {
          out = std::make_shared<ForthOutputBufferOf<int32_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int64: {
          out = std::make_shared<ForthOutputBufferOf<int64_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint8: {
          out = std::make_shared<ForthOutputBufferOf<uint8_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint16: {
          out = std::make_shared<ForthOutputBufferOf<uint16_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint32: {
          out = std::make_shared<ForthOutputBufferOf<uint32_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint64: {
          out = std::make_shared<ForthOutputBufferOf<uint64_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        // case dtype::float16: { }
        case dtype::float32: {
          out = std::make_shared<ForthOutputBufferOf<float>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::float64: {
          out = std::make_shared<ForthOutputBufferOf<double>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        // case dtype::float128: { }
        // case dtype::complex64: { }
        // case dtype::complex128: { }
        // case dtype::complex256: { }
        // case dtype::datetime64: { }
        // case dtype::timedelta64: { }
        default: {
          throw std::runtime_error("unimplemented ForthOutputBuffer type");
        }
      }
      outputs[name] = out;
      outs.push_back(out);
    }

    stack_.clear();
    for (int64_t i = 0;  i < variables_.size();  i++) {
      variables_[i] = 0;
    }

    ForthError err = ForthError::none;
    do_run(ins, outs, err);
    switch (err) {
      case ForthError::stack_underflow: {
        throw std::invalid_argument("Forth stack underflow while filling array");
      }
      case ForthError::read_beyond: {
        throw std::invalid_argument("Forth read beyond end of input while filling array");
      }
      case ForthError::seek_beyond: {
        throw std::invalid_argument("Forth seek beyond input while filling array");
      }
      case ForthError::skip_beyond: {
        throw std::invalid_argument("Forth skip beyond input while filling array");
      }
      case ForthError::rewind_beyond: {
        throw std::invalid_argument("Forth rewind beyond beginning of output while filling array");
      }
    }

    return outputs;
  }

private:
  void compile(const std::string& source) {
    std::vector<std::string> dictionary_names;
    std::vector<std::vector<I>> dictionary;

    // ...

    variable_names_.push_back("cumulative");
    variables_.push_back(0);

    input_names_.push_back("testin");
    output_names_.push_back("testout");
    output_dtypes_.push_back(dtype::int32);

    instructions_offsets_.push_back(0);
    instructions_.push_back(LITERAL);
    instructions_.push_back(1);
    instructions_.push_back(LITERAL);
    instructions_.push_back(2);
    instructions_.push_back(LITERAL);
    instructions_.push_back(3);
    instructions_.push_back(LITERAL);
    instructions_.push_back(4);
    instructions_.push_back(LITERAL);
    instructions_.push_back(5);
    instructions_offsets_.push_back(instructions_.size());
  }

  int64_t instructions_length(ForthInstructionPointer& pointer) {
    int64_t start = instructions_offsets_[pointer.which()];
    int64_t stop = instructions_offsets_[pointer.which() + 1];
    return stop - start;
  }

  I get_instruction(ForthInstructionPointer& pointer) {
    int64_t start = instructions_offsets_[pointer.which()];
    return instructions_[start + pointer.where()];
  }

  void do_run(const std::vector<std::shared_ptr<ForthInputBuffer>>& ins,
              const std::vector<std::shared_ptr<ForthOutputBuffer>>& outs,
              ForthError& err) noexcept {
    ForthInstructionPointer pointer(recursion_depth_);
    pointer.push(0, 0, 0);

    while (!pointer.empty()) {
      while (pointer.where() < instructions_length(pointer)) {
        I instruction = get_instruction(pointer);
        pointer.where() += 1;

        if (instruction < 0) {
          bool direct = instruction & PARSER_DIRECT;
          bool repeated = instruction & PARSER_REPEATED;
          bool bigendian = instruction & PARSER_BIGENDIAN;
          switch (instruction & PARSER_MASK) {
            case PARSER_BOOL: {
              break;
            }

            case PARSER_INT8: {
              break;
            }

            case PARSER_INT16: {
              break;
            }

            case PARSER_INT32: {
              break;
            }

            case PARSER_INT64: {
              break;
            }

            case PARSER_SSIZE: {
              break;
            }

            case PARSER_UINT8: {
              break;
            }

            case PARSER_UINT16: {
              break;
            }

            case PARSER_UINT32: {
              break;
            }

            case PARSER_UINT64: {
              break;
            }

            case PARSER_USIZE: {
              break;
            }

            case PARSER_FLOAT32: {
              break;
            }

            case PARSER_FLOAT64: {
              break;
            }
          }
        }
        else {
          switch (instruction) {
            case LITERAL: {
              I num = get_instruction(pointer);
              pointer.where() += 1;
              stack_.push((T)num);
              break;
            }

            case PUT: {
              break;
            }

            case INC: {
              break;
            }

            case GET: {
              break;
            }

            case SKIP: {
              break;
            }

            case SEEK: {
              break;
            }

            case END: {
              break;
            }

            case POSITION: {
              break;
            }

            case LENGTH_INPUT: {
              break;
            }

            case REWIND: {
              break;
            }

            case LENGTH_OUTPUT: {
              break;
            }

            case WRITE: {
              break;
            }

            case IF: {
              break;
            }

            case IF_ELSE: {
              break;
            }

            case DO: {
              break;
            }

            case DO_STEP: {
              break;
            }

            case AGAIN: {
              break;
            }

            case UNTIL: {
              break;
            }

            case WHILE: {
              break;
            }

            case EXIT: {
              break;
            }

            case INDEX_I: {
              break;
            }

            case INDEX_J: {
              break;
            }

            case INDEX_K: {
              break;
            }

            case DUP: {
              break;
            }

            case DROP: {
              break;
            }

            case SWAP: {
              break;
            }

            case OVER: {
              break;
            }

            case ROT: {
              break;
            }

            case NIP: {
              break;
            }

            case TUCK: {
              break;
            }

            case ADD: {
              break;
            }

            case SUB: {
              break;
            }

            case MUL: {
              break;
            }

            case DIV: {
              break;
            }

            case MOD: {
              break;
            }

            case DIVMOD: {
              break;
            }

            case LSHIFT: {
              break;
            }

            case RSHIFT: {
              break;
            }

            case ABS: {
              break;
            }

            case MIN: {
              break;
            }

            case MAX: {
              break;
            }

            case NEGATE: {
              break;
            }

            case ADD1: {
              break;
            }

            case SUB1: {
              break;
            }

            case EQ0: {
              break;
            }

            case EQ: {
              break;
            }

            case NE: {
              break;
            }

            case GT: {
              break;
            }

            case GE: {
              break;
            }

            case LT: {
              break;
            }

            case LE: {
              break;
            }

            case AND: {
              break;
            }

            case OR: {
              break;
            }

            case XOR: {
              break;
            }

            case INVERT: {
              break;
            }

            case FALSE: {
              break;
            }

            case TRUE: {
              break;
            }

            default: {
              break;
            }
          }
        }

      }

      pointer.pop();
    }
  }

  int64_t initial_buffer_;
  double resize_buffer_;

  ForthStack<T> stack_;
  int64_t recursion_depth_;

  std::vector<std::string> variable_names_;
  std::vector<T> variables_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<dtype> output_dtypes_;

  std::vector<int64_t> instructions_offsets_;
  std::vector<I> instructions_;
};


int main() {
  ForthMachine<int32_t, int32_t, true> vm("");

  std::shared_ptr<int32_t> test_input_ptr = std::shared_ptr<int32_t>(
      new int32_t[10], array_deleter<int32_t>());
  for (int64_t i = 0;  i < 10;  i++) {
    test_input_ptr.get()[i] = i % 100;
  }

  std::map<std::string, std::shared_ptr<ForthInputBuffer>> inputs;
  inputs["testin"] = std::make_shared<ForthInputBuffer>(test_input_ptr, 0, 10);

  std::map<std::string, std::shared_ptr<ForthOutputBuffer>> outputs = vm.run(inputs);
  std::cout << vm.stack().tostring() << std::endl;

  for (auto pair : outputs) {
    std::cout << pair.first << std::endl;
    std::shared_ptr<void> ptr = pair.second.get()->ptr();
    for (int64_t i = 0;  i < pair.second.get()->length();  i++) {
      std::cout << i << std::endl;
    }
  }

}

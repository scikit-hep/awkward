// c++ virtual-machine.cpp -o virtual-machine-test  &&  ./virtual-machine-test

#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstring>
#include <iostream>


// template <typename T>
// class array_deleter {
// public:
//     void operator()(T const *ptr) {
//       delete [] ptr;
//     }
// };


enum class ForthError {
  none,
  stack_underflow,
  size
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

  void resize(int64_t reservation) noexcept {
    T* new_buffer = new T[reservation];
    std::memcpy(new_buffer, buffer_, sizeof(T) * std::min(reservation, reserved_));
    delete [] buffer_;
    buffer_ = new_buffer;
    length_ = std::min(std::min(reservation, reserved_), length_);
    reserved_ = reservation;
  }

  void push(T value) noexcept {
    if (length_ == reserved_) {
      resize((int64_t)std::ceil(reserved_ * resize_));
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
               int64_t initial_stack=1024,
               double resize_stack=1.5,
               int64_t recursion_depth=1024)
    : stack_(initial_stack, resize_stack)
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

  ForthError run(/* inputs, outputs */) {
    ForthError err = ForthError::none;

    stack_.clear();

    ForthInstructionPointer pointer(recursion_depth_);
    pointer.push(0, 0, 0);

    while (!pointer.empty()) {
      while (pointer.where() < instructions_length(pointer)) {
        I instruction = get_instruction(pointer);
        pointer.where() += 1;

        if (false) {

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

    return err;
  }

private:
  void compile(const std::string& source) {
    std::vector<std::string> dictionary_names;
    std::vector<std::vector<I>> dictionary;

    // ...

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

  ForthStack<T> stack_;
  int64_t recursion_depth_;

  std::vector<std::string> variable_names_;
  std::vector<T> variables_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::vector<int64_t> instructions_offsets_;
  std::vector<I> instructions_;
};


int main() {
  ForthMachine<int32_t, int8_t, true> vm("");
  vm.run();
  std::cout << vm.stack().tostring() << std::endl;
}

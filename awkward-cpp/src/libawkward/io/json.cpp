// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/io/json.cpp", line)

#include <complex>

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/error/en.h"

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/common.h"

#include "awkward/io/json.h"

namespace rj = rapidjson;

namespace awkward {
  class Handler: public rj::BaseReaderHandler<rj::UTF8<>, Handler> {
  public:
    Handler(ArrayBuilder& builder,
            const char* nan_string,
            const char* posinf_string,
            const char* neginf_string)
        : builder_(builder)
        , moved_(false)
        , nan_string_(nan_string)
        , posinf_string_(posinf_string)
        , neginf_string_(neginf_string) { }

    void
    reset_moved() {
      moved_ = false;
    }

    bool
    moved() const {
      return moved_;
    }

    bool Null() {
      moved_ = true;
      builder_.null();
      return true;
    }

    bool Bool(bool x) {
      moved_ = true;
      builder_.boolean(x);
      return true;
    }

    bool Int(int x) {
      moved_ = true;
      builder_.integer((int64_t)x);
      return true;
    }

    bool Uint(unsigned int x) {
      moved_ = true;
      builder_.integer((int64_t)x);
      return true;
    }

    bool Int64(int64_t x) {
      moved_ = true;
      builder_.integer(x);
      return true;
    }

    bool Uint64(uint64_t x) {
      moved_ = true;
      builder_.integer((int64_t)x);
      return true;
    }

    bool Double(double x) {
      builder_.real(x);
      moved_ = true;
      return true;
    }

    bool
    String(const char* str, rj::SizeType length, bool /* copy */) {
      moved_ = true;
      if (nan_string_ != nullptr  &&  strcmp(str, nan_string_) == 0) {
        builder_.real(std::numeric_limits<double>::quiet_NaN());
        return true;
      }
      else if (posinf_string_ != nullptr  &&  strcmp(str, posinf_string_) == 0) {
        builder_.real(std::numeric_limits<double>::infinity());
        return true;
      }
      else if (neginf_string_ != nullptr  &&  strcmp(str, neginf_string_) == 0) {
        builder_.real(-std::numeric_limits<double>::infinity());
        return true;
      }
      else {
        builder_.string(str, (int64_t)length);
        return true;
      }
    }

    bool
    StartArray() {
      moved_ = true;
      builder_.beginlist();
      return true;
    }

    bool
    EndArray(rj::SizeType /* numfields */) {
      moved_ = true;
      builder_.endlist();
      return true;
    }

    bool
    StartObject() {
      moved_ = true;
      builder_.beginrecord();
      return true;
    }

    bool
    EndObject(rj::SizeType /* numfields */) {
      moved_ = true;
      builder_.endrecord();
      return true;
    }

    bool
    Key(const char* str, rj::SizeType /* length */, bool /* copy */) {
      moved_ = true;
      builder_.field_check(str);
      return true;
    }

  private:
    ArrayBuilder& builder_;
    bool moved_;
    const char* nan_string_;
    const char* posinf_string_;
    const char* neginf_string_;
  };

  class FileLikeObjectStream {
  public:
    typedef char Ch;

    FileLikeObjectStream(FileLikeObject* source, int64_t buffersize)
      : source_(source)
      , buffersize_(buffersize)
      , bufferlast_(0)
      , current_(0)
      , readcount_(0)
      , count_(0)
      , eof_(false) {
      buffer_ = new char[(size_t)buffersize];
      read();
    }

    ~FileLikeObjectStream() {
      delete [] buffer_;
    }

    Ch Peek() const {
      return *current_;
     }
    Ch Take() {
      Ch c = *current_;
      read();
      return c;
    }
    size_t Tell() const {
      return (size_t)count_ + static_cast<size_t>(current_ - buffer_);
    }

    std::string error_context() const {
      int64_t current = (int64_t)current_ - (int64_t)buffer_;
      int64_t bufferafter = (int64_t)bufferlast_ - (int64_t)buffer_;
      if (*bufferlast_ != 0) {
        bufferafter++;
      }

      int64_t start = current - 40;
      if (start < 0) {
        start = 0;
      }
      int64_t stop = current + 20;
      if (stop > bufferafter) {
        stop = bufferafter;
      }

      std::string context = std::string(buffer_, (size_t)stop).substr((size_t)start);
      size_t arrow = (size_t)(current - start);

      size_t pos;

      pos = 0;
      while ((size_t)(pos = context.find(9, pos)) != std::string::npos) {
        context.replace(pos, 1, "\\t");
        pos++;
        if (pos < arrow) {
          arrow++;
        }
      }

      pos = 0;
      while ((size_t)(pos = context.find(10, pos)) != std::string::npos) {
        context.replace(pos, 1, "\\n");
        pos++;
        if (pos < arrow) {
          arrow++;
        }
      }

      pos = 0;
      while ((size_t)(pos = context.find(13, pos)) != std::string::npos) {
        context.replace(pos, 1, "\\r");
        pos++;
        if (pos < arrow) {
          arrow++;
        }
      }

      return std::string("\nJSON: ") + context + std::string("\n") + std::string(arrow + 6, '-') + "^";
    }

    // not implemented
    void Put(Ch) { assert(false); }
    void Flush() { assert(false); }
    Ch* PutBegin() { assert(false); return 0; }
    size_t PutEnd(Ch*) { assert(false); return 0; }

  private:
    void read() {
      if (current_ < bufferlast_) {
        ++current_;
      }
      else if (!eof_) {
        count_ += readcount_;
        readcount_ = source_->read(buffersize_, buffer_);
        bufferlast_ = buffer_ + readcount_ - 1;
        current_ = buffer_;

        if (readcount_ < buffersize_) {
          buffer_[readcount_] = '\0';
          ++bufferlast_;
          eof_ = true;
        }
      }
    }

    FileLikeObject* source_;
    int64_t buffersize_;
    Ch* buffer_;
    Ch* bufferlast_;
    Ch* current_;
    int64_t readcount_;
    int64_t count_;
    bool eof_;
  };

  void
  fromjsonobject(FileLikeObject* source,
                 ArrayBuilder& builder,
                 int64_t buffersize,
                 bool read_one,
                 const char* nan_string,
                 const char* posinf_string,
                 const char* neginf_string) {

    rj::Reader reader;
    FileLikeObjectStream stream(source, buffersize);
    Handler handler(builder,
                    nan_string,
                    posinf_string,
                    neginf_string);

    if (read_one) {
      bool fully_parsed = reader.Parse(stream, handler);
      if (!fully_parsed) {
        throw std::invalid_argument(
          std::string("JSON syntax error at char ")
          + std::to_string(stream.Tell())
          + std::string("\n")
          + stream.error_context()
          + FILENAME(__LINE__));
      }
    }

    else {
      while (stream.Peek() != 0) {
        handler.reset_moved();
        bool fully_parsed = reader.Parse<rj::kParseStopWhenDoneFlag>(stream, handler);
        if (handler.moved()) {
          if (!fully_parsed) {
            if (stream.Peek() == 0) {
              throw std::invalid_argument(
                  std::string("incomplete JSON object at the end of the stream")
                  + std::string("\n")
                  + stream.error_context()
                  + FILENAME(__LINE__));
            }
            else {
              throw std::invalid_argument(
                std::string("JSON syntax error at char ")
                + std::to_string(stream.Tell())
                + std::string("\n")
                + stream.error_context()
                + FILENAME(__LINE__));
            }
          }
        }
        else if (stream.Peek() != 0) {
          throw std::invalid_argument(
            std::string("JSON syntax error at char ")
            + std::to_string(stream.Tell())
            + std::string("\n")
            + stream.error_context()
            + FILENAME(__LINE__));
        }
      }
    }

  }

  #define TopLevelArray 0           // no arguments
  #define FillByteMaskedArray 1     // arg1: ByteMaskedArray output
  #define FillIndexedOptionArray 2  // arg1: IndexedOptionArray output, arg2: counter
  #define FillBoolean 3             // arg1: boolean output
  #define FillInteger 4             // arg1: integer output
  #define FillNumber 5              // arg1: number output
  #define FillString 6              // arg1: offsets output, arg2: content output
  #define FillEnumString 7          // arg1: index output, arg2: strings start, arg3: strings stop
  #define FillNullEnumString 8      // arg1: index output, arg2: strings start, arg3: strings stop
  #define VarLengthList 9           // arg1: offsets output
  #define FixedLengthList 10        // arg1: expected length
  #define KeyTableHeader 11         // arg1: number of items, arg2: record identifier
  #define KeyTableItem 12           // arg1: string index, arg2: jump to instruction

  class HandlerSchema: public rj::BaseReaderHandler<rj::UTF8<>, HandlerSchema> {
  public:
    HandlerSchema(FromJsonObjectSchema* specializedjson,
                  const char* nan_string,
                  const char* posinf_string,
                  const char* neginf_string)
      : specializedjson_(specializedjson)
      , nan_string_(nan_string)
      , posinf_string_(posinf_string)
      , neginf_string_(neginf_string)
      , moved_(false)
      , schema_okay_(true)
      , ignore_(0) { }

    void
    reset_moved() {
      moved_ = false;
    }

    bool
    moved() const {
      return moved_;
    }

    bool
    schema_failure() const {
      return !schema_okay_;
    }

    bool Null() {
      moved_ = true;
      // std::cout << "null " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 0);
          specializedjson_->step_forward();

          // std::cout << "  FillByteMaskedArray " << specializedjson_->debug() << std::endl;

          switch (specializedjson_->instruction()) {
            case FillBoolean:
              specializedjson_->write_int8(specializedjson_->argument1(), 0);
              break;
            case FillInteger:
              specializedjson_->write_int64(specializedjson_->argument1(), 0);
              break;
            case FillNumber:
              specializedjson_->write_float64(specializedjson_->argument1(), 0.0);
              break;
            case FillString:
              specializedjson_->write_add_int64(specializedjson_->argument1(), 0);
              break;
            case VarLengthList:
              specializedjson_->write_add_int64(specializedjson_->argument1(), 0);
              break;
            default:
              return schema_okay_ = false;
          }
          specializedjson_->step_backward();
          return true;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(specializedjson_->argument1(), -1);
          return true;
        case FillNullEnumString:
          specializedjson_->write_int64(specializedjson_->argument1(), -1);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Bool(bool x) {
      moved_ = true;
      // std::cout << "bool " << x << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Bool(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Bool(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillBoolean:
          specializedjson_->write_int8(specializedjson_->argument1(), x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Int(int x) {
      moved_ = true;
      // std::cout << "int " << x << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Int(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Int(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Uint(unsigned int x) {
      moved_ = true;
      // std::cout << "uint " << x << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Uint(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Uint(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Int64(int64_t x) {
      moved_ = true;
      // std::cout << "int64 " << x << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Int64(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Int64(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), (double)x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Uint64(uint64_t x) {
      moved_ = true;
      // std::cout << "uint64 " << x << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Uint64(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Uint64(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), (int64_t)x);
          return true;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), (double)x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Double(double x) {
      moved_ = true;
      // std::cout << "double " << x << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Double(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Double(x);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillInteger:
          return schema_okay_ = false;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool
    String(const char* str, rj::SizeType length, bool copy) {
      moved_ = true;
      // std::cout << "string " << str << " " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      int64_t enumi;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = String(str, length, copy);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = String(str, length, copy);
          specializedjson_->step_backward();
          return schema_okay_ = out;
        case FillString:
          specializedjson_->write_add_int64(specializedjson_->argument1(), length);
          specializedjson_->write_many_uint8(
            specializedjson_->argument2(), length, reinterpret_cast<const uint8_t*>(str)
          );
          return true;
        case FillEnumString:
        case FillNullEnumString:
          enumi = specializedjson_->find_enum(str);
          if (enumi == -1) {
            return schema_okay_ = false;
          }
          else {
            specializedjson_->write_int64(specializedjson_->argument1(), enumi);
            return true;
          }
        case FillNumber:
          if (nan_string_ != nullptr  &&  strcmp(str, nan_string_) == 0) {
            specializedjson_->write_float64(
              specializedjson_->argument1(), std::numeric_limits<double>::quiet_NaN()
            );
            return true;
          }
          else if (posinf_string_ != nullptr  &&  strcmp(str, posinf_string_) == 0) {
            specializedjson_->write_float64(
              specializedjson_->argument1(), std::numeric_limits<double>::infinity()
            );
            return true;
          }
          else if (neginf_string_ != nullptr  &&  strcmp(str, neginf_string_) == 0) {
            specializedjson_->write_float64(
              specializedjson_->argument1(), -std::numeric_limits<double>::infinity()
            );
            return true;
          }
          else {
            return schema_okay_ = false;
          }
        default:
          return schema_okay_ = false;
      }
    }

    bool
    StartArray() {
      moved_ = true;
      // std::cout << "startarray " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        ignore_++;
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      switch (specializedjson_->instruction()) {
        case TopLevelArray:
          specializedjson_->push_stack(specializedjson_->current_instruction() + 1);
          return true;
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->push_stack(specializedjson_->current_instruction() + 2);
          return true;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->push_stack(specializedjson_->current_instruction() + 2);
          return true;
        case VarLengthList:
          specializedjson_->push_stack(specializedjson_->current_instruction() + 1);
          return true;
        case FixedLengthList:
          specializedjson_->push_stack(specializedjson_->current_instruction() + 1);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool
    EndArray(rj::SizeType numfields) {
      moved_ = true;
      // std::cout << "endarray " << specializedjson_->debug() << std::endl;
      // std::cout << "  ignore state " << ignore_ << std::endl;

      if (ignore_) {
        ignore_--;
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      bool out;
      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case TopLevelArray:
          specializedjson_->add_to_length(numfields);
          return true;
        case FillByteMaskedArray:
        case FillIndexedOptionArray:
          specializedjson_->step_forward();
          switch (specializedjson_->instruction()) {
            case VarLengthList:
              specializedjson_->write_add_int64(specializedjson_->argument1(), numfields);
              out = true;
              break;
            case FixedLengthList:
              out = numfields == specializedjson_->argument1();
              break;
            default:
              return schema_okay_ = false;
          }
          specializedjson_->step_backward();
          return out;
        case VarLengthList:
          specializedjson_->write_add_int64(specializedjson_->argument1(), numfields);
          return true;
        case FixedLengthList:
          return numfields == specializedjson_->argument1();
        default:
          return schema_okay_ = false;
      }
    }

    bool
    StartObject() {
      moved_ = true;
      // std::cout << "startobject " << specializedjson_->debug() << std::endl;

      if (ignore_) {
        ignore_++;
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->start_object(specializedjson_->current_instruction() + 1);
          specializedjson_->push_stack(specializedjson_->current_instruction() + 1);
          return true;
        case KeyTableHeader:
          specializedjson_->start_object(specializedjson_->current_instruction());
          specializedjson_->push_stack(specializedjson_->current_instruction());
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool
    EndObject(rj::SizeType /* numfields */) {
      moved_ = true;
      // std::cout << "endobject " << specializedjson_->debug() << std::endl;
      // std::cout << "  ignore state " << ignore_ << std::endl;

      if (ignore_ == 1) {
        // we have ignored a single previous key, now reach EndObject
        ignore_--;
      }
      if (ignore_) {
        ignore_--;
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          if (!specializedjson_->end_object(specializedjson_->current_instruction() + 1)) {
            return nulls_for_optiontype();
          }
          return true;
        case KeyTableHeader:
          if (!specializedjson_->end_object(specializedjson_->current_instruction())) {
            return nulls_for_optiontype();
          }
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool
    nulls_for_optiontype() {
      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          specializedjson_->push_stack(specializedjson_->current_instruction() + 1);
        case KeyTableHeader:
          specializedjson_->push_stack(specializedjson_->current_instruction());
      }
      int64_t keytableheader_instruction = specializedjson_->current_instruction();
      int64_t num_fields = specializedjson_->argument1();
      int64_t record_identifier = specializedjson_->argument2();
      specializedjson_->pop_stack();

      // for each not-already-filled key, fill it if it's option-type, error otherwise
      for (int64_t i = keytableheader_instruction + 1;  i <= keytableheader_instruction + num_fields;  i++) {
        int64_t j = i - (keytableheader_instruction + 1);
        if (!specializedjson_->key_already_filled(record_identifier, j)) {
          int64_t jump_to = specializedjson_->key_instruction_at(i);

          specializedjson_->push_stack(jump_to);
          switch (specializedjson_->instruction()) {
            case FillByteMaskedArray:
            case FillIndexedOptionArray:
            case FillNullEnumString:
              Null();
              break;
            default:
              schema_okay_ = false;
          }
          specializedjson_->pop_stack();

          if (!schema_okay_) {
            return false;
          }
        }
      }

      return true;
    }

    bool
    Key(const char* str, rj::SizeType /* length */, bool /* copy */) {
      moved_ = true;
      // std::cout << "key " << str << " " << specializedjson_->debug() << std::endl;
      // std::cout << "  ignore state " << ignore_ << std::endl;

      if (ignore_ == 1) {
        // we have ignored a single previous key, now reach a new Key
        ignore_--;
      }
      if (ignore_) {
        // std::cout << "  ignoring!" << std::endl;
        return true;
      }

      int64_t jump_to;
      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          specializedjson_->step_forward();
          jump_to = specializedjson_->find_key(str);
          specializedjson_->step_backward();
          if (jump_to == -1) {
            ignore_ = 1;
          }
          // jump_to might be -1, but it will be popped by the next Key or EndObject
          specializedjson_->push_stack(jump_to);
          return true;
        case KeyTableHeader:
          jump_to = specializedjson_->find_key(str);
          if (jump_to == -1) {
            ignore_ = 1;
          }
          // jump_to might be -1, but it will be popped by the next Key or EndObject
          specializedjson_->push_stack(jump_to);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

  private:
    FromJsonObjectSchema* specializedjson_;
    const char* nan_string_;
    const char* posinf_string_;
    const char* neginf_string_;
    bool moved_;
    bool schema_okay_;

    // if 0, read data; otherwise, ignore data
    // if 1, ignore this key, but go to ignore_ = 0 for the next Key/EndObject
    // StartArray/StartObject increases ignore_ by 1
    // EndArray/EndObject decrease ignore_ by 1
    int64_t ignore_;
  };

  FromJsonObjectSchema::FromJsonObjectSchema(FileLikeObject* source,
                                             int64_t buffersize,
                                             bool read_one,
                                             const char* nan_string,
                                             const char* posinf_string,
                                             const char* neginf_string,
                                             const char* jsonassembly,
                                             int64_t initial,
                                             double resize) {
    BuilderOptions options(initial, resize);

    rj::Document doc;
    doc.Parse<rj::kParseDefaultFlags>(jsonassembly);

    if (doc.HasParseError()) {
      throw std::invalid_argument(
        "failed to parse jsonassembly" + FILENAME(__LINE__)
      );
    }

    if (!doc.IsArray()) {
      throw std::invalid_argument(
        "jsonassembly must be an array of instructions" + FILENAME(__LINE__)
      );
    }

    int64_t instruction_stack_max_depth = 0;
    std::vector<std::string> strings;
    bool is_record = true;

    for (auto& item : doc.GetArray()) {
      if (!item.IsArray()  ||  item.Size() == 0  ||  !item[0].IsString()) {
        throw std::invalid_argument(
          "each jsonassembly instruction must be an array starting with a string" +
          FILENAME(__LINE__)
        );
      }

      if (std::string("TopLevelArray") == item[0].GetString()) {
        if (item.Size() != 1) {
          throw std::invalid_argument(
            "TopLevelArray arguments: (none!)" + FILENAME(__LINE__)
          );
        }
        instruction_stack_max_depth++;
        instructions_.push_back(TopLevelArray);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
        instructions_.push_back(-1);

        is_record = false;
      }

      else if (std::string("FillByteMaskedArray") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillByteMaskedArray arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        if (std::string("int8") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillByteMaskedArray argument 2 (dtype:str) must be 'int8'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int8);
        int64_t outi = (int64_t)buffers_uint8_.size();
        output_which_.push_back(outi);
        buffers_uint8_.push_back(GrowableBuffer<uint8_t>(options));
        instructions_.push_back(FillByteMaskedArray);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }

      else if (std::string("FillIndexedOptionArray") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillIndexedOptionArray arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        if (std::string("int64") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillIndexedOptionArray argument 2 (dtype:str) must be 'int64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int64);
        int64_t outi = (int64_t)buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        int64_t counti = (int64_t)counters_.size();
        counters_.push_back(0);
        instructions_.push_back(FillIndexedOptionArray);
        instructions_.push_back(outi);
        instructions_.push_back(counti);
        instructions_.push_back(-1);
      }

      else if (std::string("FillBoolean") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillBoolean arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        if (std::string("uint8") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillBoolean argument 2 (dtype:str) must be 'uint8'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::uint8);
        int64_t outi = (int64_t)buffers_uint8_.size();
        output_which_.push_back(outi);
        buffers_uint8_.push_back(GrowableBuffer<uint8_t>(options));
        instructions_.push_back(FillBoolean);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }

      else if (std::string("FillInteger") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillInteger arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        if (std::string("int64") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillInteger argument 2 (dtype:str) must be 'int64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int64);
        int64_t outi = (int64_t)buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        instructions_.push_back(FillInteger);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }

      else if (std::string("FillNumber") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillNumber arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        if (std::string("float64") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillNumber argument 2 (dtype:str) must be 'float64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::float64);
        int64_t outi = (int64_t)buffers_float64_.size();
        output_which_.push_back(outi);
        buffers_float64_.push_back(GrowableBuffer<double>(options));
        instructions_.push_back(FillNumber);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }

      else if (std::string("FillString") == item[0].GetString()) {
        if (item.Size() != 5  ||  !item[1].IsString()  ||  !item[2].IsString()  ||  !item[3].IsString()  ||  !item[4].IsString()) {
          throw std::invalid_argument(
            "FillString arguments: offsets:str offsets_dtype:str content:str content_dtype:str" + FILENAME(__LINE__)
          );
        }
        if (std::string("int64") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillString argument 2 (dtype:str) must be 'int64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int64);
        int64_t offsetsi = (int64_t)buffers_int64_.size();
        output_which_.push_back(offsetsi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        buffers_int64_[(size_t)offsetsi].append(0);
        if (std::string("uint8") != item[4].GetString()) {
          throw std::invalid_argument(
            "FillString argument 4 (dtype:str) must be 'uint8'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[3].GetString());
        output_dtypes_.push_back(util::dtype::uint8);
        int64_t contenti = (int64_t)buffers_uint8_.size();
        output_which_.push_back(contenti);
        buffers_uint8_.push_back(GrowableBuffer<uint8_t>(options));
        instructions_.push_back(FillString);
        instructions_.push_back(offsetsi);
        instructions_.push_back(contenti);
        instructions_.push_back(-1);
      }

      else if (std::string("FillEnumString") == item[0].GetString()  ||
               std::string("FillNullEnumString") == item[0].GetString()) {
        if (item.Size() != 4  ||  !item[1].IsString()  ||  !item[2].IsString()  ||  !item[3].IsArray()) {
          throw std::invalid_argument(
            "FillEnumString/FillNullEnumString arguments: index:str dtype:str [strings]" +
            FILENAME(__LINE__)
          );
        }
        if (std::string("int64") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillEnumString/FillNullEnumString argument 2 (dtype:str) must be 'int64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int64);
        int64_t outi = (int64_t)buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        int64_t start = (int64_t)strings.size();
        for (auto& x : item[3].GetArray()) {
          if (!x.IsString()) {
            throw std::invalid_argument(
              "FillEnumString/FillNullEnumString list of strings (argument 3) must all be strings" +
              FILENAME(__LINE__)
            );
          }
          strings.push_back(x.GetString());
        }
        int64_t stop = (int64_t)strings.size();
        if (std::string("FillEnumString") == item[0].GetString()) {
          instructions_.push_back(FillEnumString);
        }
        else {
          instructions_.push_back(FillNullEnumString);
        }
        instructions_.push_back(outi);
        instructions_.push_back(start);
        instructions_.push_back(stop);
      }

      else if (std::string("VarLengthList") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "VarLengthList arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        instruction_stack_max_depth++;


        if (std::string("int64") != item[2].GetString()) {
          throw std::invalid_argument(
            "VarLengthList argument 2 (dtype:str) must be 'int64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int64);
        int64_t outi = (int64_t)buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        buffers_int64_[(size_t)outi].append(0);
        instructions_.push_back(VarLengthList);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }

      else if (std::string("FixedLengthList") == item[0].GetString()) {
        if (item.Size() != 2  ||  !item[1].IsInt64()) {
          throw std::invalid_argument(
            "FixedLengthList arguments: length:int" + FILENAME(__LINE__)
          );
        }
        instruction_stack_max_depth++;
        instructions_.push_back(FixedLengthList);
        instructions_.push_back(item[1].GetInt64());
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }

      else if (std::string("KeyTableHeader") == item[0].GetString()) {
        if (item.Size() != 2  ||  !item[1].IsInt64()) {
          throw std::invalid_argument(
            "KeyTableHeader arguments: num_items:int" + FILENAME(__LINE__)
          );
        }
        int64_t num_items = item[1].GetInt64();
        int64_t num_checklist_chunks = num_items / 64;
        if (num_items % 64 != 0) {
          num_checklist_chunks++;
        }

        instruction_stack_max_depth++;
        instructions_.push_back(KeyTableHeader);
        instructions_.push_back(num_items);
        instructions_.push_back(record_current_field_.size());  // record identifier
        instructions_.push_back(-1);

        record_current_field_.push_back(-1);  // the first find_key will increase this to zero

        // checklist consists of a 1 bit for each field that has not been seen yet
        std::vector<uint64_t> checklist_init(num_checklist_chunks, 0);
        for (int64_t j = 0;  j < num_items;  j++) {
          int64_t  chunki    = j >> 6;                      // j / 64
          uint64_t chunkmask = (uint64_t)1 << (j & 0x3f);   // j % 64
          checklist_init[chunki] |= chunkmask;
        }
        std::vector<uint64_t> checklist_copy = checklist_init;   // copied (not shared)
        record_checklist_init_.push_back(checklist_init);
        record_checklist_.push_back(checklist_copy);
      }

      else if (std::string("KeyTableItem") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsInt64()) {
          throw std::invalid_argument(
            "KeyTableItem arguments: key:str jump_to:int" + FILENAME(__LINE__)
          );
        }
        int64_t stringi = (int64_t)strings.size();
        strings.push_back(item[1].GetString());
        instructions_.push_back(KeyTableItem);
        instructions_.push_back(stringi);
        instructions_.push_back(item[2].GetInt64());
        instructions_.push_back(-1);
      }

      else {
        throw std::invalid_argument(
          std::string("unrecognized jsonassembly instruction: ") + item[0].GetString() +
          FILENAME(__LINE__)
        );
      }
    }

    string_offsets_.push_back(0);
    for (auto string : strings) {
      string_offsets_.push_back(string_offsets_[string_offsets_.size() - 1] + (int64_t)string.length());
      for (auto c : string) {
        characters_.push_back(c);
      }
    }

    for (int64_t i = 0;  i < instruction_stack_max_depth;  i++) {
      instruction_stack_.push_back(-1);
    }

    current_instruction_ = 0;
    current_stack_depth_ = 0;
    length_ = 0;

    rj::Reader reader;
    FileLikeObjectStream stream(source, buffersize);
    HandlerSchema handler(this,
                          nan_string,
                          posinf_string,
                          neginf_string);

    if (read_one) {
      bool fully_parsed = reader.Parse(stream, handler);
      if (!fully_parsed) {
        std::string reason(handler.schema_failure() ? "JSON schema mismatch before char " : "JSON syntax error at char ");
        throw std::invalid_argument(
          reason
          + std::to_string(stream.Tell())
          + std::string("\n")
          + stream.error_context()
          + FILENAME(__LINE__));
      }
      if (is_record) {
        length_ = 1;
      }
    }

    else {
      while (stream.Peek() != 0) {
        handler.reset_moved();
        bool fully_parsed = reader.Parse<rj::kParseStopWhenDoneFlag>(stream, handler);
        if (handler.moved()) {
          if (!fully_parsed) {
            if (stream.Peek() == 0) {
              throw std::invalid_argument(
                  std::string("incomplete JSON object at the end of the stream")
                  + std::string("\n")
                  + stream.error_context()
                  + FILENAME(__LINE__));
            }
            else {
              std::string reason(handler.schema_failure() ? "JSON schema mismatch before char " : "JSON syntax error at char ");
              throw std::invalid_argument(
                reason
                + std::to_string(stream.Tell())
                + std::string("\n")
                + stream.error_context()
                + FILENAME(__LINE__));
            }
          }
          if (is_record) {
            length_++;
          }
        }
        else if (stream.Peek() != 0) {
          std::string reason(handler.schema_failure() ? "JSON schema mismatch before char " : "JSON syntax error at char ");
          throw std::invalid_argument(
            reason
            + std::to_string(stream.Tell())
            + std::string("\n")
            + stream.error_context()
            + FILENAME(__LINE__));
        }
      }
    }

  }

  std::string
  FromJsonObjectSchema::debug() const noexcept {
    std::stringstream out;
    out << "at " << current_instruction_ << " | " << instructions_[(size_t)current_instruction_ * 4] << " stack";
    for (int64_t i = 0;  (size_t)i < instruction_stack_.size();  i++) {
      if (i == current_stack_depth_) {
        out << " ;";
      }
      out << " " << instruction_stack_.data()[i];
    }
    if ((size_t)current_stack_depth_ == instruction_stack_.size()) {
      out << " ;";
    }
    return out.str();
  }

}

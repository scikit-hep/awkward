// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/SpecializedJSON.cpp", line)

#include <stdexcept>

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

#include "awkward/forth/SpecializedJSON.h"

namespace rj = rapidjson;

namespace awkward {
  #define FillByteMaskedArray 0     // arg1: ByteMaskedArray output
  #define FillIndexedOptionArray 1  // arg1: IndexedOptionArray output, arg2: counter
  #define FillBoolean 2             // arg1: boolean output
  #define FillInteger 3             // arg1: integer output
  #define FillNumber 4              // arg1: number output
  #define FillString 5              // arg1: offsets output, arg2: content output
  #define FillEnumString 6          // arg1: index output, arg2: strings start, arg3: strings stop
  #define VarLengthList 7           // arg1: offsets output
  #define FixedLengthList 8         // no arguments
  #define KeyTableHeader 9          // arg1: number of items
  #define KeyTableItem 10           // arg1: string index, arg2: jump to instruction

  class Handler: public rj::BaseReaderHandler<rj::UTF8<>, Handler> {
  public:
    Handler(SpecializedJSON* specializedjson): specializedjson_(specializedjson) { }

    bool Null() {
      std::cout << "Null instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool Bool(bool x) {
      std::cout << "Bool " << x << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool Int(int x) {
      std::cout << "Int " << x << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool Uint(unsigned int x) {
      std::cout << "Uint " << x << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool Int64(int64_t x) {
      std::cout << "Int64 " << x << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool Uint64(uint64_t x) {
      std::cout << "Uint64 " << x << " instruction: " << specializedjson_->instruction() << std::endl;
      return false;
    }

    bool Double(double x) {
      std::cout << "Double " << x << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool
    String(const char* str, rj::SizeType length, bool copy) {
      std::cout << "String " << str << " " << length << " " << copy << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool
    StartArray() {
      std::cout << "StartArray instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          specializedjson_->step_forward();
          return true;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool
    EndArray(rj::SizeType numfields) {
      std::cout << "EndArray " << numfields << " instruction: " << specializedjson_->instruction() << std::endl;
      specializedjson_->step_backward();
      return true;
    }

    bool
    StartObject() {
      std::cout << "StartObject instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool
    EndObject(rj::SizeType numfields) {
      std::cout << "EndObject " << numfields << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

    bool
    Key(const char* str, rj::SizeType length, bool copy) {
      std::cout << "Key " << str << " " << length << " " << copy << " instruction: " << specializedjson_->instruction() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          return false;
        case FillIndexedOptionArray:
          return false;
        case FillBoolean:
          return false;
        case FillInteger:
          return false;
        case FillNumber:
          return false;
        case FillString:
          return false;
        case FillEnumString:
          return false;
        case VarLengthList:
          return false;
        case FixedLengthList:
          return false;
        case KeyTableHeader:
          return false;
        case KeyTableItem:
          return false;
      }
      return false;
    }

  private:
    SpecializedJSON* specializedjson_;
  };

  SpecializedJSON::SpecializedJSON(const std::string& jsonassembly,
                                   int64_t output_initial_size,
                                   double output_resize_factor) {
    rj::Document doc;
    doc.Parse<rj::kParseDefaultFlags>(jsonassembly.c_str());

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

    for (auto& item : doc.GetArray()) {
      if (!item.IsArray()  ||  item.Size() == 0  ||  !item[0].IsString()) {
        throw std::invalid_argument(
          "each jsonassembly instruction must be an array starting with a string" +
          FILENAME(__LINE__)
        );
      }

      if (std::string("FillByteMaskedArray") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillByteMaskedArray arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    false,
                                    output_initial_size,
                                    output_resize_factor);
        instructions_.push_back(FillByteMaskedArray);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }
      else if (std::string("FillIndexedOptionArray") == item[0].GetString()) {
        instructions_.push_back(FillIndexedOptionArray);
      }
      else if (std::string("FillBoolean") == item[0].GetString()) {
        instructions_.push_back(FillBoolean);
      }
      else if (std::string("FillInteger") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillInteger arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    false,
                                    output_initial_size,
                                    output_resize_factor);
        instructions_.push_back(FillInteger);
        instructions_.push_back(outi);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }
      else if (std::string("FillNumber") == item[0].GetString()) {
        instructions_.push_back(FillNumber);
      }
      else if (std::string("FillString") == item[0].GetString()) {
        instructions_.push_back(FillString);
      }
      else if (std::string("FillEnumString") == item[0].GetString()) {
        instructions_.push_back(FillEnumString);
      }
      else if (std::string("VarLengthList") == item[0].GetString()) {
        instructions_.push_back(VarLengthList);
      }
      else if (std::string("FixedLengthList") == item[0].GetString()) {
        if (item.Size() != 1) {
          throw std::invalid_argument(
            "FixedLengthList arguments: (none!)" + FILENAME(__LINE__)
          );
        }
        instructions_.push_back(FixedLengthList);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }
      else if (std::string("KeyTableHeader") == item[0].GetString()) {
        instructions_.push_back(KeyTableHeader);
      }
      else if (std::string("KeyTableItem") == item[0].GetString()) {
        instructions_.push_back(KeyTableItem);
      }
      else {
        throw std::invalid_argument(
          std::string("unrecognized jsonassembly instruction: ") + item[0].GetString() +
          FILENAME(__LINE__)
        );
      }
    }

    for (int64_t i = 0;  i < instruction_stack_max_depth;  i++) {
      instruction_stack_.push_back(0);
    }

    std::cout << "num instructions " << instructions_.size() << std::endl;
  }

  const std::shared_ptr<ForthOutputBuffer>
  SpecializedJSON::output_at(const std::string& name) const {
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return outputs_[i];
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  util::dtype
  SpecializedJSON::dtype_at(const std::string& name) const {
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (output_names_[i] == name) {
        return output_dtypes_[i];
      }
    }
    throw std::invalid_argument(
      std::string("output not found: ") + name + FILENAME(__LINE__)
    );
  }

  bool
  SpecializedJSON::parse_string(const char* source) noexcept {
    reset();
    rj::Reader reader;
    rj::StringStream stream(source);
    Handler handler(this);
    bool out = reader.Parse<rj::kParseDefaultFlags>(stream, handler);
    json_position_ = stream.Tell();
    return out;
  }

  void
  SpecializedJSON::reset() noexcept {
    current_instruction_ = 0;
    current_stack_depth_ = 0;
    for (int64_t i = 0;  i < counters_.size();  i++) {
      counters_[i] = 0;
    }
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      outputs_[i].get()->reset();
      if (output_leading_zero_[i]) {
        outputs_[i].get()->write_one_int64(0, false);
      }
    }
  }

  int64_t
  SpecializedJSON::output_index(const std::string& name,
                                util::dtype dtype,
                                bool leading_zero,
                                int64_t init,
                                double resize) {
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      if (name == output_names_[i]) {
        if (dtype != output_dtypes_[i]  ||  leading_zero != output_leading_zero_[i]) {
          throw std::invalid_argument(
            std::string("redeclaration of ") + name +
            std::string(" with a different dtype or leading zero") + FILENAME(__LINE__)
          );
        }
        return i;
      }
    }
    int64_t i = output_names_.size();
    output_names_.push_back(name);
    output_dtypes_.push_back(dtype);
    output_leading_zero_.push_back(leading_zero);

    std::shared_ptr<ForthOutputBuffer> out;
    switch (dtype) {
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
    outputs_.push_back(out);

    return i;
  }

}

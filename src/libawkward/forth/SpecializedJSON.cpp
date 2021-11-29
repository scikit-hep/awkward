// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/SpecializedJSON.cpp", line)

#include <stdexcept>
#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

#include "awkward/forth/SpecializedJSON.h"

namespace rj = rapidjson;

namespace awkward {
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
  #define KeyTableHeader 11         // arg1: number of items
  #define KeyTableItem 12           // arg1: string index, arg2: jump to instruction

  class SpecializedJSONHandler: public rj::BaseReaderHandler<rj::UTF8<>, SpecializedJSONHandler> {
  public:
    SpecializedJSONHandler(SpecializedJSON* specializedjson): specializedjson_(specializedjson) { }

    bool Null() {
      // std::cout << "null " << specializedjson_->debug() << std::endl;

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
              return false;
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
          return false;
      }
    }

    bool Bool(bool x) {
      // std::cout << "bool " << x << " " << specializedjson_->debug() << std::endl;

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Bool(x);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Bool(x);
          specializedjson_->step_backward();
          return out;
        case FillBoolean:
          specializedjson_->write_int8(specializedjson_->argument1(), x);
          return true;
        default:
          return false;
      }
    }

    bool Int(int x) {
      // std::cout << "int " << x << " " << specializedjson_->debug() << std::endl;

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Int(x);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Int(x);
          specializedjson_->step_backward();
          return out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        default:
          return false;
      }
    }

    bool Uint(unsigned int x) {
      // std::cout << "uint " << x << " " << specializedjson_->debug() << std::endl;

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Uint(x);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Uint(x);
          specializedjson_->step_backward();
          return out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        default:
          return false;
      }
    }

    bool Int64(int64_t x) {
      // std::cout << "int64 " << x << " " << specializedjson_->debug() << std::endl;

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Int64(x);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Int64(x);
          specializedjson_->step_backward();
          return out;
        case FillInteger:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        default:
          return false;
      }
    }

    bool Uint64(uint64_t x) {
      // std::cout << "uint64 " << x << " " << specializedjson_->debug() << std::endl;

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Uint64(x);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Uint64(x);
          specializedjson_->step_backward();
          return out;
        case FillInteger:
          specializedjson_->write_uint64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_uint64(specializedjson_->argument1(), x);
          return true;
        default:
          return false;
      }
    }

    bool Double(double x) {
      // std::cout << "double " << x << " " << specializedjson_->debug() << std::endl;

      bool out;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = Double(x);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = Double(x);
          specializedjson_->step_backward();
          return out;
        case FillInteger:
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        default:
          return false;
      }
    }

    bool
    String(const char* str, rj::SizeType length, bool copy) {
      // std::cout << "string " << str << " " << specializedjson_->debug() << std::endl;

      bool out;
      int64_t enumi;
      switch (specializedjson_->instruction()) {
        case FillByteMaskedArray:
          specializedjson_->write_int8(specializedjson_->argument1(), 1);
          specializedjson_->step_forward();
          out = String(str, length, copy);
          specializedjson_->step_backward();
          return out;
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->step_forward();
          out = String(str, length, copy);
          specializedjson_->step_backward();
          return out;
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
            return false;
          }
          else {
            specializedjson_->write_int64(specializedjson_->argument1(), enumi);
            return true;
          }
        default:
          return false;
      }
    }

    bool
    StartArray() {
      // std::cout << "startarray " << specializedjson_->debug() << std::endl;

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
          return false;
      }
    }

    bool
    EndArray(rj::SizeType numfields) {
      // std::cout << "endarray " << specializedjson_->debug() << std::endl;

      bool out;
      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case TopLevelArray:
          specializedjson_->set_length(numfields);
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
              return false;
          }
          specializedjson_->step_backward();
          return out;
        case VarLengthList:
          specializedjson_->write_add_int64(specializedjson_->argument1(), numfields);
          return true;
        case FixedLengthList:
          return numfields == specializedjson_->argument1();
        default:
          return false;
      }
    }

    bool
    StartObject() {
      // std::cout << "startobject " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          specializedjson_->write_int64(
            specializedjson_->argument1(),
            specializedjson_->get_and_increment(specializedjson_->argument2())
          );
          specializedjson_->push_stack(specializedjson_->current_instruction() + 1);
          return true;
        case KeyTableHeader:
          specializedjson_->push_stack(specializedjson_->current_instruction());
          return true;
        default:
          return false;
      }
    }

    bool
    EndObject(rj::SizeType numfields) {
      // std::cout << "endobject " << specializedjson_->debug() << std::endl;

      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          return true;
        case KeyTableHeader:
          return true;
        default:
          return false;
      }
    }

    bool
    Key(const char* str, rj::SizeType length, bool copy) {
      // std::cout << "key " << specializedjson_->debug() << std::endl;

      int64_t jump_to;
      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          specializedjson_->step_forward();
          jump_to = specializedjson_->find_key(str);
          if (jump_to == -1) {
            return false;
          }
          else {
            specializedjson_->step_backward();
            specializedjson_->push_stack(jump_to);
            return true;
          }
        case KeyTableHeader:
          jump_to = specializedjson_->find_key(str);
          if (jump_to == -1) {
            return false;
          }
          else {
            specializedjson_->push_stack(jump_to);
            return true;
          }
        default:
          return false;
      }
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
      }

      else if (std::string("FillByteMaskedArray") == item[0].GetString()) {
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
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillIndexedOptionArray arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    false,
                                    output_initial_size,
                                    output_resize_factor);
        int64_t counti = counters_.size();
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
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    false,
                                    output_initial_size,
                                    output_resize_factor);
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
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsString()) {
          throw std::invalid_argument(
            "FillNumber arguments: output:str dtype:str" + FILENAME(__LINE__)
          );
        }
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    false,
                                    output_initial_size,
                                    output_resize_factor);
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
        int64_t offsetsi = output_index(item[1].GetString(),
                                        util::name_to_dtype(item[2].GetString()),
                                        true,
                                        output_initial_size,
                                        output_resize_factor);
        int64_t contenti = output_index(item[3].GetString(),
                                        util::name_to_dtype(item[4].GetString()),
                                        false,
                                        output_initial_size,
                                        output_resize_factor);
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
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    false,
                                    output_initial_size,
                                    output_resize_factor);
        int64_t start = strings.size();
        for (auto& x : item[3].GetArray()) {
          if (!x.IsString()) {
            throw std::invalid_argument(
              "FillEnumString/FillNullEnumString list of strings (argument 3) must all be strings" +
              FILENAME(__LINE__)
            );
          }
          strings.push_back(x.GetString());
        }
        int64_t stop = strings.size();
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
        int64_t outi = output_index(item[1].GetString(),
                                    util::name_to_dtype(item[2].GetString()),
                                    true,
                                    output_initial_size,
                                    output_resize_factor);
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
        instruction_stack_max_depth++;
        instructions_.push_back(KeyTableHeader);
        instructions_.push_back(item[1].GetInt64());
        instructions_.push_back(-1);
        instructions_.push_back(-1);
      }
      else if (std::string("KeyTableItem") == item[0].GetString()) {
        if (item.Size() != 3  ||  !item[1].IsString()  ||  !item[2].IsInt64()) {
          throw std::invalid_argument(
            "KeyTableItem arguments: key:str jump_to:int" + FILENAME(__LINE__)
          );
        }
        int64_t stringi = strings.size();
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

    for (int64_t i = 0;  i < instruction_stack_max_depth;  i++) {
      instruction_stack_.push_back(-1);
    }

    string_offsets_.push_back(0);
    for (auto string : strings) {
      string_offsets_.push_back(string_offsets_[string_offsets_.size() - 1] + string.length());
      for (auto c : string) {
        characters_.push_back(c);
      }
    }

    // std::cout << debug_listing();
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

  int64_t
  SpecializedJSON::length() const noexcept {
    return length_;
  }

  int64_t
  SpecializedJSON::json_position() const noexcept {
    return json_position_;
  }

  bool
  SpecializedJSON::parse_string(const char* source) noexcept {
    reset();
    rj::Reader reader;
    rj::StringStream stream(source);
    SpecializedJSONHandler handler(this);
    bool out = reader.Parse<rj::kParseDefaultFlags>(stream, handler);
    json_position_ = stream.Tell();
    return out;
  }

  std::string
  SpecializedJSON::debug() const noexcept {
    std::stringstream out;
    out << "at " << current_instruction_ << " | " << instructions_[current_instruction_ * 4] << " stack";
    for (int64_t i = 0;  i < instruction_stack_.size();  i++) {
      if (i == current_stack_depth_) {
        out << " ;";
      }
      out << " " << instruction_stack_.data()[i];
    }
    if (current_stack_depth_ == instruction_stack_.size()) {
      out << " ;";
    }
    return out.str();
  }

  std::string
  SpecializedJSON::debug_listing() const noexcept {
    std::stringstream out;
    for (int64_t i = 0;  i < instructions_.size() / 4;  i++) {
      out << i << " | " << instructions_[i * 4];
      switch (instructions_[i * 4]) {
        case TopLevelArray:
          out << " TopLevelArray ";
          break;
        case FillByteMaskedArray:
          out << " FillByteMaskedArray ";
          break;
        case FillIndexedOptionArray:
          out << " FillIndexedOptionArray ";
          break;
        case FillBoolean:
          out << " FillBoolean ";
          break;
        case FillInteger:
          out << " FillInteger ";
          break;
        case FillNumber:
          out << " FillNumber ";
          break;
        case FillString:
          out << " FillString ";
          break;
        case FillEnumString:
          out << " FillEnumString ";
          break;
        case FillNullEnumString:
          out << " FillNullEnumString ";
          break;
        case VarLengthList:
          out << " VarLengthList ";
          break;
        case FixedLengthList:
          out << " FixedLengthList ";
          break;
        case KeyTableHeader:
          out << " KeyTableHeader ";
          break;
        case KeyTableItem:
          out << " KeyTableItem ";
          break;
        default:
          out << " ??? ";
          break;
      }
      out << instructions_[i * 4 + 1] << " "
          << instructions_[i * 4 + 2] << " "
          << instructions_[i * 4 + 3] << std::endl;
    }
    return out.str();
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
    length_ = 0;
  }

  int64_t
  SpecializedJSON::output_index(const std::string& name,
                                util::dtype dtype,
                                bool leading_zero,
                                int64_t init,
                                double resize) {
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

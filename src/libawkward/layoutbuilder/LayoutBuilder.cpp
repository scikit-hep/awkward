// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/LayoutBuilder.cpp", line)

#include "awkward/layoutbuilder/LayoutBuilder.h"

#include "awkward/layoutbuilder/BitMaskedArrayBuilder.h"
#include "awkward/layoutbuilder/ByteMaskedArrayBuilder.h"
#include "awkward/layoutbuilder/EmptyArrayBuilder.h"
#include "awkward/layoutbuilder/IndexedArrayBuilder.h"
#include "awkward/layoutbuilder/IndexedOptionArrayBuilder.h"
#include "awkward/layoutbuilder/ListArrayBuilder.h"
#include "awkward/layoutbuilder/ListOffsetArrayBuilder.h"
#include "awkward/layoutbuilder/NumpyArrayBuilder.h"
#include "awkward/layoutbuilder/RecordArrayBuilder.h"
#include "awkward/layoutbuilder/RegularArrayBuilder.h"
#include "awkward/layoutbuilder/UnionArrayBuilder.h"
#include "awkward/layoutbuilder/UnmaskedArrayBuilder.h"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"

namespace rj = rapidjson;

namespace awkward {

  const std::string
  index_form_to_name(const std::string& form_index) {
    if (form_index == "i8") {
      return "int8";
    }
    if (form_index == "u8") {
      return "uint8";
    }
    if (form_index == "i32") {
      return "int32";
    }
    if (form_index == "u32") {
      return "uint32";
    }
    if (form_index == "i64") {
      return "int64";
    }
    throw std::runtime_error(
      std::string("unrecognized Index::Form ") + FILENAME(__LINE__));

  }

  const std::string
  index_form_to_vm_format(const std::string& form_index) {
    if (form_index == "i8") {
      return "b";
    }
    if (form_index == "u8") {
      return "B";
    }
    if (form_index == "i32") {
      return "i";
    }
    if (form_index == "u32") {
      return "I";
    }
    if (form_index == "i64") {
      return "q";
    }
    throw std::runtime_error(
      std::string("unrecognized Index::Form ") + FILENAME(__LINE__));
  }

  const std::string
  primitive_to_state(const std::string& name) {
    if (name == "bool") {
      return std::to_string(static_cast<utype>(state::boolean));
    }
    else if (name == "int8") {
      return std::to_string(static_cast<utype>(state::int8));
    }
    else if (name == "int16") {
      return std::to_string(static_cast<utype>(state::int16));
    }
    else if (name == "int32") {
      return std::to_string(static_cast<utype>(state::int32));
    }
    else if (name == "int64") {
      return std::to_string(static_cast<utype>(state::int64));
    }
    else if (name == "uint8") {
      return std::to_string(static_cast<utype>(state::uint8));
    }
    else if (name == "uint16") {
      return std::to_string(static_cast<utype>(state::uint16));
    }
    else if (name == "uint32") {
      return std::to_string(static_cast<utype>(state::uint32));
    }
    else if (name == "uint64") {
      return std::to_string(static_cast<utype>(state::uint64));
    }
    else if (name == "float16") {
      return std::to_string(static_cast<utype>(state::float16));
    }
    else if (name == "float32") {
      return std::to_string(static_cast<utype>(state::float32));
    }
    else if (name == "float64") {
      return std::to_string(static_cast<utype>(state::float64));
    }
    else if (name == "float128") {
      return std::to_string(static_cast<utype>(state::float128));
    }
    else if (name == "complex64") {
      return std::to_string(static_cast<utype>(state::complex64));
    }
    else if (name == "complex128") {
      return std::to_string(static_cast<utype>(state::complex128));
    }
    else if (name == "complex256") {
      return std::to_string(static_cast<utype>(state::complex256));
    }
    else if (name.rfind("datetime64", 0) == 0) {
      return std::to_string(static_cast<utype>(state::datetime64));
    }
    else if (name.rfind("timedelta64", 0) == 0) {
      return std::to_string(static_cast<utype>(state::timedelta64));
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized util::dtype ") + FILENAME(__LINE__));
    }
  };

  const std::string
  primitive_to_vm_format(const std::string& name) {
    if (name == "bool") {
      return "?";
    }
    else if (name == "int8") {
      return "b";
    }
    else if (name == "int16") {
      return "h";
    }
    else if (name == "int32") {
      return "i";
    }
    else if (name == "int64") {
      return "q";
    }
    else if (name == "uint8") {
      return "B";
    }
    else if (name == "uint16") {
      return "H";
    }
    else if (name == "uint32") {
      return "I";
    }
    else if (name == "uint64") {
      return "Q";
    }
    else if ((name == "float16")  ||
             (name == "float32")) {
      return "f";
    }
    else if ((name == "float64")  ||
             (name == "float128")  ||
             (name == "complex64")  ||
             (name == "complex128")  ||
             (name == "complex256")) {
      return "d";
    }
    else if (name.rfind("datetime64", 0) == 0) {
      return "M";
    }
    else if (name.rfind("timedelta64", 0) == 0) {
      return "m";
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized util::dtype ") + FILENAME(__LINE__));
    }
  };

  template <typename T, typename I>
  int64_t LayoutBuilder<T, I>::next_node_id = 0;
  template <typename T, typename I>
  int64_t LayoutBuilder<T, I>::error_id = 0;

  template <typename T, typename I>
  LayoutBuilder<T, I>::LayoutBuilder(const std::string& json_form,
                                     const int64_t initial,
                                     bool vm_init)
    : json_form_(json_form),
      initial_(initial),
      builder_(nullptr),
      vm_(nullptr),
      vm_input_data_("data"),
      vm_source_() {
    LayoutBuilder<T, I>::error_id = 0;
    vm_source_ = std::string("variable err ");
    vm_source_.append("input ")
      .append(vm_input_data_).append(" ");

    initialise_builder(json_form);

    vm_source_.append(builder_.get()->vm_error()).append(" ");
    vm_source_.append(builder_.get()->vm_output()).append(" ");
    vm_source_.append(builder_.get()->vm_func()).append(" ");
    vm_source_.append(builder_.get()->vm_from_stack()).append(" ");

    vm_source_.append("0 begin pause ")
      .append(builder_.get()->vm_func_name())
      .append(" 1+ again ");

    if (vm_init) {
      initialise();
    }
  }

  template <typename T, typename I>
  const std::string
  LayoutBuilder<T, I>::to_buffers(BuffersContainer& container) const {
    return builder_.get()->to_buffers(container, vm().get()->outputs());
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::initialise_builder(const std::string& json_form) {
    try {
      builder_ = form_builder_from_json(json_form);
    }
    catch (const std::exception& e) {
      throw std::invalid_argument(
        std::string("builder initialization failed:\n\t")
        + e.what() + std::string(" ")
        + FILENAME(__LINE__));
    }
  }

  template <typename T, typename I>
  template <typename JSON>
  FormBuilderPtr<T, I>
  LayoutBuilder<T, I>::from_json(const JSON& json_doc) {

    if (json_doc.IsString()) {
      std::string primitive = json_doc.GetString();
      std::string json_form_key = std::string("node")
        + std::to_string(LayoutBuilder<T, I>::next_id());

      return std::make_shared<NumpyArrayBuilder<T, I>>(util::Parameters(),
                                                       json_form_key,
                                                       primitive,
                                                       primitive_to_state(primitive),
                                                       primitive_to_vm_format(primitive));
    }

    std::string json_form_key;
    std::string json_form_index;
    std::string json_form_offsets;

    if (json_doc.HasMember("form_key")) {
      if (json_doc["form_key"].IsNull()) {
        json_form_key = std::string("node")
          + std::to_string(LayoutBuilder<T, I>::next_id());
      }
      else if (json_doc["form_key"].IsString()) {
        json_form_key = json_doc["form_key"].GetString();
      }
      else {
        throw std::invalid_argument(
          std::string("'form_key' must be null or a string") + FILENAME(__LINE__));
      }
    }
    else {
      json_form_key = std::string("node")
        + std::to_string(LayoutBuilder<T, I>::next_id());
    }

    if (json_doc.IsObject()  &&
        json_doc.HasMember("class")  &&
        json_doc["class"].IsString()) {

      util::Parameters json_form_parameters;
      if (json_doc.HasMember("parameters")) {
        if (json_doc["parameters"].IsObject()) {
          for (auto& pair : json_doc["parameters"].GetObject()) {
            rj::StringBuffer stringbuffer;
            rj::Writer<rj::StringBuffer> writer(stringbuffer);
            pair.value.Accept(writer);
            json_form_parameters[pair.name.GetString()] = stringbuffer.GetString();
          }
        }
        else {
          throw std::invalid_argument(
            std::string("'parameters' must be a JSON object") + FILENAME(__LINE__));
        }
      }
      std::string cls = json_doc["class"].GetString();

      if (cls == std::string("BitMaskedArray")) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }

        return std::make_shared<BitMaskedArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                             json_form_parameters,
                                                             json_form_key);
      }
      if (cls == std::string("ByteMaskedArray")) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }

        return std::make_shared<ByteMaskedArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                              json_form_parameters,
                                                              json_form_key);
      }
      if (cls == std::string("EmptyArray")) {
        return std::make_shared<EmptyArrayBuilder<T, I>>(json_form_parameters);
      }

      if ((cls == std::string("IndexedArray"))  ||
          (cls == std::string("IndexedArray64"))  ||
          (cls == std::string("IndexedArrayU32"))  ||
          (cls == std::string("IndexedArray32"))) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }

        bool is_categorical(false);
        if (util::parameter_equals(json_form_parameters, "__array__", "\"categorical\"")) {
          is_categorical = true;
        }
        if (json_doc.HasMember("index")  &&  json_doc["index"].IsString()) {
          json_form_index = json_doc["index"].GetString();
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'index' specification")
            + FILENAME(__LINE__));
        }

        return std::make_shared<IndexedArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                           json_form_parameters,
                                                           json_form_key,
                                                           index_form_to_name(json_form_index),
                                                           is_categorical);
      }

      if ((cls == std::string("IndexedOptionArray"))  ||
          (cls == std::string("IndexedOptionArray64"))  ||
          (cls == std::string("IndexedOptionArray32"))) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }

        bool is_categorical(false);
        if (util::parameter_equals(json_form_parameters, "__array__", "\"categorical\"")) {
          is_categorical = true;
        }
        if (json_doc.HasMember("index")  &&  json_doc["index"].IsString()) {
          json_form_index = json_doc["index"].GetString();
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'index' specification")
            + FILENAME(__LINE__));
        }

        return std::make_shared<IndexedOptionArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                                 json_form_parameters,
                                                                 json_form_key,
                                                                 index_form_to_name(json_form_index),
                                                                 is_categorical);
      }

      if ((cls == std::string("ListArray"))  ||
          (cls == std::string("ListArray64")) ||
          (cls == std::string("ListArrayU32"))  ||
          (cls == std::string("ListArray32"))) {

        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }

        std::string json_form_starts;
        if (json_doc.HasMember("starts")  &&  json_doc["starts"].IsString()) {
          json_form_starts = json_doc["stops"].GetString();
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'starts' specification")
            + FILENAME(__LINE__));
        }

        return std::make_shared<ListArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                        json_form_parameters,
                                                        json_form_key,
                                                        index_form_to_name(json_form_starts));
      }

      if ((cls == std::string("ListOffsetArray"))  ||
          (cls == std::string("ListOffsetArray64"))  ||
          (cls == std::string("ListOffsetArrayU32"))  ||
          (cls == std::string("ListOffsetArray32"))) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        bool is_string_builder(false);
        if (util::parameter_equals(json_form_parameters, "__array__", "\"string\"")  ||
            util::parameter_equals(json_form_parameters, "__array__", "\"bytestring\"")) {
          is_string_builder = true;
        }

        if (json_doc.HasMember("offsets")  &&  json_doc["offsets"].IsString()) {
          json_form_offsets = json_doc["offsets"].GetString();
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" is missing an 'offsets' specification")
            + FILENAME(__LINE__));
        }

        return std::make_shared<ListOffsetArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                              json_form_parameters,
                                                              json_form_key,
                                                              index_form_to_name(json_form_offsets),
                                                              is_string_builder);
      }
      if (cls == std::string("NumpyArray")) {
        std::string primitive;

        if (json_doc.HasMember("primitive")  &&  json_doc["primitive"].IsString()) {
          primitive = json_doc["primitive"].GetString();
         }
         else {
           throw std::invalid_argument(
             std::string("NumpyForm must have a 'primitive' field")
                         + FILENAME(__LINE__));
         }
        return std::make_shared<NumpyArrayBuilder<T, I>>(json_form_parameters,
                                                         json_form_key,
                                                         primitive,
                                                         primitive_to_state(primitive),
                                                         primitive_to_vm_format(primitive));
      }
      if (cls == std::string("RecordArray")) {
        util::RecordLookupPtr recordlookup(nullptr);
        std::vector<FormBuilderPtr<T, I>> contents;
        if (json_doc.HasMember("contents")  &&  json_doc["contents"].IsArray()) {
          for (auto& x : json_doc["contents"].GetArray()) {
            contents.push_back(from_json(x));
          }
        }
        else if (json_doc.HasMember("contents")  &&  json_doc["contents"].IsObject()) {
          recordlookup = std::make_shared<util::RecordLookup>();
          for (auto& pair : json_doc["contents"].GetObject()) {
            recordlookup.get()->push_back(pair.name.GetString());
            contents.push_back(from_json(pair.value));
          }
        }
        else {
          throw std::invalid_argument(
            std::string("RecordArray 'contents' must be a JSON list or a "
                        "JSON object") + FILENAME(__LINE__));
        }
        return std::make_shared<RecordArrayBuilder<T, I>>(contents,
                                                          recordlookup,
                                                          json_form_parameters,
                                                          json_form_key);
      }
      if (cls == std::string("RegularArray")) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }
        if (!json_doc.HasMember("size")  ||  !json_doc["size"].IsInt()) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'size'") + FILENAME(__LINE__));
        }
        int64_t json_form_size = json_doc["size"].GetInt64();
        return std::make_shared<RegularArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                           json_form_parameters,
                                                           json_form_key,
                                                           json_form_size);
      }

      if ((cls == std::string("UnionArray"))  ||
          (cls == std::string("UnionArray8_64"))  ||
          (cls == std::string("UnionArray8_U32"))  ||
          (cls == std::string("UnionArray8_32"))) {
        std::string json_form_tags;
        if (json_doc.HasMember("tags")  &&  json_doc["tags"].IsString()) {
          json_form_tags = json_doc["tags"].GetString();
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'tags' specification")
            + FILENAME(__LINE__));
        }
        std::string json_form_index;
        if (json_doc.HasMember("index")  &&  json_doc["index"].IsString()) {
          json_form_index = json_doc["index"].GetString();
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" is missing a 'index' specification")
            + FILENAME(__LINE__));
        }
        std::vector<FormBuilderPtr<T, I>> contents;
        if (json_doc.HasMember("contents")  &&  json_doc["contents"].IsArray()) {
          for (auto& x : json_doc["contents"].GetArray()) {
            contents.push_back(from_json(x));
          }
        }
        else {
          throw std::invalid_argument(
            cls + std::string(" 'contents' must be a JSON list ")
            + FILENAME(__LINE__));
        }

        return std::make_shared<UnionArrayBuilder<T, I>>(contents,
                                                         json_form_parameters,
                                                         json_form_key,
                                                         json_form_tags,
                                                         json_form_index);
      }
      if (cls == std::string("UnmaskedArray")) {
        if (!json_doc.HasMember("content")) {
          throw std::invalid_argument(
            cls + std::string(" is missing its 'content'") + FILENAME(__LINE__));
        }

        return std::make_shared<UnmaskedArrayBuilder<T, I>>(from_json(json_doc["content"]),
                                                            json_form_parameters,
                                                            json_form_key);
      }
      throw std::invalid_argument(
        std::string("LayoutBuilder does not recognise the Form ")
        + FILENAME(__LINE__));
    }

    rj::StringBuffer stringbuffer;
    rj::PrettyWriter<rj::StringBuffer> writer(stringbuffer);
    json_doc.Accept(writer);
    throw std::invalid_argument(
            std::string("JSON cannot be recognized as a Form:\n")
            + stringbuffer.GetString() + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  FormBuilderPtr<T, I>
  LayoutBuilder<T, I>::form_builder_from_json(const std::string& json_form) {
    rj::Document json_doc;
    json_doc.Parse<rj::kParseNanAndInfFlag>(json_form.c_str());

    if (json_doc.IsObject()) {
      return from_json(json_doc);
    }

    throw std::invalid_argument(
            std::string("JSON cannot be recognized as a Form:\n")
            + json_form + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::connect(const std::shared_ptr<ForthMachineOf<T, I>>& vm) {
    if (vm_ == nullptr) {
      vm_ = vm;

      std::shared_ptr<void> ptr(
        kernel::malloc<void>(kernel::lib::cpu, 8*sizeof(uint8_t)));

      vm_inputs_map_[vm_input_data_] = std::make_shared<ForthInputBuffer>(ptr, 0, 8);
      vm_.get()->run(vm_inputs_map_);
    }
    else {
      throw std::invalid_argument(
        std::string("LayoutBuilder is already connected to a Virtual Machine ")
        + FILENAME(__LINE__));
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::initialise() {
    vm_ = std::make_shared<ForthMachineOf<T, I>>(vm_source());

    std::shared_ptr<void> ptr(
      kernel::malloc<void>(kernel::lib::cpu, initial_*(int64_t)sizeof(uint8_t)));

    vm_inputs_map_[vm_input_data_] = std::make_shared<ForthInputBuffer>(ptr, 0, initial_);
    vm_.get()->run(vm_inputs_map_);
  }

  template <typename T, typename I>
  template <typename D>
  void
  LayoutBuilder<T, I>::set_data(D x) {
    reinterpret_cast<D*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = x;
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::resume() const {
    if (vm_.get()->resume() == util::ForthError::user_halt) {
      throw std::invalid_argument(vm_.get()->string_at(vm_.get()->stack().back()));
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::debug_step() const {
    std::cout << "stack ";
    for (auto const& i : vm_.get()->stack()) {
      std::cout << i << ", ";
    }
    std::cout << "\n";
    for (auto const& i : vm_.get()->outputs()) {
      std::cout << i.first << " : ";
      std::cout << i.second.get()->toNumpyArray().get()->tostring();
      std::cout << "\n";
    }
  }

  template <typename T, typename I>
  const std::string
  LayoutBuilder<T, I>::vm_source() const {
    return vm_source_;
  }

  template <typename T, typename I>
  const std::shared_ptr<ForthMachineOf<T, I>>
  LayoutBuilder<T, I>::vm() const {
    if (vm_ != nullptr) {
      return vm_;
    }
    else {
      throw std::invalid_argument(
        std::string("LayoutBuilder is not connected to a Virtual Machine ")
        + FILENAME(__LINE__));
    }
  }

  template <typename T, typename I>
  int64_t
  LayoutBuilder<T, I>::next_id() {
    return LayoutBuilder<T, I>::next_node_id++;
  }

  template <typename T, typename I>
  int64_t
  LayoutBuilder<T, I>::next_error_id() {
    return LayoutBuilder<T, I>::error_id++;
  }

  template <typename T, typename I>
  int64_t
  LayoutBuilder<T, I>::length() const {
    return builder_->len(vm().get()->outputs());
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::pre_snapshot() const {
    vm().get()->maybe_throw(util::ForthError::user_halt, ignore_);
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::null() {
    vm_.get()->stack_push(static_cast<utype>(state::null));
    resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::boolean(bool x) {
    builder_.get()->boolean(x, this);
  }

  template<typename T, typename I>
  void
  LayoutBuilder<T, I>::add_bool(bool x) {
    set_data<bool>(x);
    vm_.get()->stack_push(static_cast<utype>(state::boolean));
    resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::int64(int64_t x) {
    if (vm_.get()->is_ready()) {
      builder_.get()->int64(x, this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template<typename T, typename I>
  void
  LayoutBuilder<T, I>::add_int64(int64_t x) {
    set_data<int64_t>(x);
    vm_.get()->stack_push(static_cast<utype>(state::int64));
    resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::float64(double x) {
    if (vm_.get()->is_ready()) {
      builder_.get()->float64(x, this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template<typename T, typename I>
  void
  LayoutBuilder<T, I>::add_double(double x) {
    set_data<double>(x);
    vm_.get()->stack_push(static_cast<utype>(state::float64));
    resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::complex(std::complex<double> x) {
    if (vm_.get()->is_ready()) {
      builder_.get()->complex(x, this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template<typename T, typename I>
  void
  LayoutBuilder<T, I>::add_complex(std::complex<double> x) {
    set_data<std::complex<double>>(x.real());
    vm_.get()->stack_push(static_cast<utype>(state::complex128));
    resume();
    set_data<std::complex<double>>(x.imag());
    vm_.get()->stack_push(static_cast<utype>(state::complex128));
    resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::bytestring(const char* x) {
    throw std::runtime_error(
      std::string("LayoutBuilder a null terminated 'bytestring' is not implemented yet")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::bytestring(const char* x, int64_t length) {
    for (int64_t i = 0; i < length; i++) {
      set_data<uint8_t>((uint8_t)x[i]);
      vm_.get()->stack_push(static_cast<utype>(state::uint8));
      resume();
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::bytestring(const std::string& x) {
    if (vm_.get()->is_ready()) {
      builder_.get()->bytestring(x, this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::string(const char* x) {
    throw std::runtime_error(
      std::string("LayoutBuilder a null terminated 'string' is not implemented yet")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::string(const char* x, int64_t length) {
    for (int64_t i = 0; i < length; i++) {
      set_data<uint8_t>((uint8_t)x[i]);
      vm_.get()->stack_push(static_cast<utype>(state::uint8));
      resume();
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::string(const std::string& x) {
    if (vm_.get()->is_ready()) {
      builder_.get()->string(x, this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template<typename T, typename I>
  void
  LayoutBuilder<T, I>::add_string(const std::string& x) {
    begin_list();
    string(x.c_str(), (int64_t)x.length());
    end_list();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::begin_list() {
    if (vm_.get()->is_ready()) {
      builder_.get()->begin_list(this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::add_begin_list() {
    vm_.get()->stack_push(static_cast<utype>(state::begin_list));
    vm_.get()->resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::end_list() {
    if (vm_.get()->is_ready()) {
      builder_.get()->end_list(this);
    }
    else {
      throw std::invalid_argument(
        "Virtual Machine has been halted; "
        "the last user error was: "
        + vm_.get()->string_at(vm_.get()->stack().back())
        + FILENAME(__LINE__)
      );
    }
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::add_end_list() {
    vm_.get()->stack_push(static_cast<utype>(state::end_list));
    vm_.get()->resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::index(int64_t x) {
    vm_.get()->stack_push((int32_t)x);
    vm_.get()->stack_push(static_cast<utype>(state::index));
    vm_.get()->resume();
  }

  template <typename T, typename I>
  void
  LayoutBuilder<T, I>::tag(int8_t x) {
    set_data<int8_t>(x);
    vm_.get()->stack_push(static_cast<utype>(state::tag));
    vm_.get()->resume();
  }

  template class EXPORT_TEMPLATE_INST LayoutBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST LayoutBuilder<int64_t, int32_t>;

}

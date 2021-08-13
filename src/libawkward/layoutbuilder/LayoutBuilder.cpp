// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/LayoutBuilder.cpp", line)

#include "awkward/layoutbuilder/LayoutBuilder.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/type/Type.h"

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
  index_form_to_name(Index::Form form) {
    switch (form) {
    case Index::Form::i8:
      return "int8";
    case Index::Form::u8:
      return "uint8";
    case Index::Form::i32:
      return "int32";
    case Index::Form::u32:
      return "uint32";
    case Index::Form::i64:
      return "int64";
    default:
      throw std::runtime_error(
        std::string("unrecognized Index::Form ") + FILENAME(__LINE__));
    }
  }

  const std::string
  index_form_to_vm_format(Index::Form form) {
    switch (form) {
    case Index::Form::i8:
      return "b";
    case Index::Form::u8:
      return "B";
    case Index::Form::i32:
      return "i";
    case Index::Form::u32:
      return "I";
    case Index::Form::i64:
      return "q";
    default:
      throw std::runtime_error(
        std::string("unrecognized Index::Form ") + FILENAME(__LINE__));
    }
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

  int64_t LayoutBuilder::next_node_id = 0;
  int64_t LayoutBuilder::error_id = 0;

  LayoutBuilder::LayoutBuilder(const std::string& json_form,
                               const ArrayBuilderOptions& options,
                               bool vm_init)
    : initial_(options.initial()),
      length_(8),
      builder_(formBuilderFromJson(json_form)),
      vm_(nullptr),
      vm_input_data_("data"),
      vm_source_() {
    LayoutBuilder::error_id = 0;
    vm_source_ = std::string("variable err").append("\n");
    vm_source_.append("input ")
      .append(vm_input_data_).append("\n");

    vm_source_.append(builder_.get()->vm_error()).append("\n");
    vm_source_.append(builder_.get()->vm_output()).append("\n");
    vm_source_.append(builder_.get()->vm_func()).append("\n");
    vm_source_.append(builder_.get()->vm_from_stack()).append("\n");

    vm_source_.append("0").append("\n")
      .append("begin").append("\n")
      .append("pause").append("\n")
      .append(builder_.get()->vm_func_name()).append("\n")
      .append("1+").append("\n")
      .append("again").append("\n");

    if (vm_init) {
      initialise();
    }
  }

  FormBuilderPtr
  LayoutBuilder::formBuilderFromJson(const std::string& json_form) {
    rj::Document json_doc;
    json_doc.Parse<rj::kParseNanAndInfFlag>(json_form.c_str());

    std::string json_form_key;
    std::string json_form_content;
    std::string json_form_index;
    std::string json_form_offsets;

    if (json_doc.HasMember("form_key")) {
      if (json_doc["form_key"].IsNull()) {
        json_form_key = std::string("node-id")
          + std::to_string(LayoutBuilder::next_id());
      }
      else if (json_doc["form_key"].IsString()) {
        json_form_key = json_doc["form_key"].GetString();
      }
      else {
        throw std::invalid_argument(
          std::string("'form_key' must be null or a string") + FILENAME(__LINE__));
      }
    }

    bool isgen;
    bool is64;
    bool isU32;
    bool is32;

    if (json_doc.IsObject()  &&
        json_doc.HasMember("class")  &&
        json_doc["class"].IsString()) {

      std::string cls = json_doc["class"].GetString();

      if (cls == std::string("BitMaskedArray")) {
        return std::make_shared<BitMaskedArrayBuilder>(json_form_key,
                                                       json_form_content);
      }
      if (cls == std::string("ByteMaskedArray")) {
        return std::make_shared<ByteMaskedArrayBuilder>(json_form_key,
                                                        json_form_content);
      }
      if (cls == std::string("EmptyArray")) {
        return std::make_shared<EmptyArrayBuilder>();
      }

      isgen = is64 = isU32 = is32 = false;
      if ((cls == std::string("IndexedArray"))  ||
               (cls == std::string("IndexedArray64"))  ||
               (cls == std::string("IndexedArrayU32"))  ||
               (cls == std::string("IndexedArray32"))) {

        bool is_categorical(false);
        //form.get()->parameter_equals("__array__", "\"categorical\"")),

//        index_form_to_name(Index::Form form) {

        return std::make_shared<IndexedArrayBuilder>(json_form_key,
                                                     json_form_index,
                                                     json_form_content,
                                                     is_categorical);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((cls == std::string("IndexedOptionArray"))  ||
               (cls == std::string("IndexedOptionArray64"))  ||
               (cls == std::string("IndexedOptionArray32"))) {

//        index_form_to_name(Index::Form form) {
        bool is_categorical(false);

        // form_.get()->parameter_equals("__array__", "\"categorical\"")) {

        return std::make_shared<IndexedOptionArrayBuilder>(json_form_key,
                                                           json_form_index,
                                                           json_form_content,
                                                           is_categorical);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((cls == std::string("ListArray"))  ||
               (cls == std::string("ListArray64")) ||
               (cls == std::string("ListArrayU32"))  ||
               (cls == std::string("ListArray32"))) {

//        index_form_to_name(Index::Form form) {
        std::string json_form_starts;

        return std::make_shared<ListArrayBuilder>(json_form_key,
                                                  json_form_starts,
                                                  json_form_content);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((cls == std::string("ListOffsetArray"))  ||
               (cls == std::string("ListOffsetArray64"))  ||
               (cls == std::string("ListOffsetArrayU32"))  ||
               (cls == std::string("ListOffsetArray32"))) {
                 // form.get()->parameter_equals("__array__", "\"string\"")
                 //     ||  form.get()->parameter_equals("__array__", "\"bytestring\"")),
        bool is_string_builder(false);

        // index_form_to_name(form_offsets)

        return std::make_shared<ListOffsetArrayBuilder>(json_form_key,
                                                        json_form_offsets,
                                                        json_form_content,
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
        return std::make_shared<NumpyArrayBuilder>(json_form_key,
                                                   primitive,
                                                   primitive_to_state(primitive),
                                                   primitive_to_vm_format(primitive));
      }
      if (cls == std::string("RecordArray")) {
        //contents_size_((int64_t)form.get()->contents().size())
        // FIXME:
        const std::vector<const std::string> json_form_contents;
        return std::make_shared<RecordArrayBuilder>(json_form_key, json_form_contents);
      }
      if (cls == std::string("RegularArray")) {
        return std::make_shared<RegularArrayBuilder>(json_form_key, json_form_content);
      }

      isgen = is64 = isU32 = is32 = false;
      if ((cls == std::string("UnionArray"))  ||
               (cls == std::string("UnionArray8_64"))  ||
               (cls == std::string("UnionArray8_U32"))  ||
               (cls == std::string("UnionArray8_32"))) {
                 // index_form_to_name(form_.get()->tags()
//                 const std::vector<const std::string>& form_contents,
        // FIXME:
        const std::vector<const std::string> json_form_contents;
// index_form_to_name(form_tags)
        std::string json_form_tags;

        return std::make_shared<UnionArrayBuilder>(json_form_key, json_form_tags, json_form_contents);
      }
      if (cls == std::string("UnmaskedArray")) {
        return std::make_shared<UnmaskedArrayBuilder>(json_form, json_form_key);
      }
      throw std::invalid_argument(
        std::string("LayoutBuilder does not recognise the Form ")
        + FILENAME(__LINE__));
    }

    rj::StringBuffer stringbuffer;
    rj::PrettyWriter<rj::StringBuffer> writer(stringbuffer);
    json_doc.Accept(writer);
    throw std::invalid_argument(
            std::string("JSON cannot be recognized as a Form:\n\n")
            + stringbuffer.GetString() + FILENAME(__LINE__));
  }

  void
  LayoutBuilder::connect(const std::shared_ptr<ForthMachine32>& vm) {
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

  void
  LayoutBuilder::initialise() {
    vm_ = std::make_shared<ForthMachine32>(vm_source());

    std::shared_ptr<void> ptr(
      kernel::malloc<void>(kernel::lib::cpu, initial_*(int64_t)sizeof(uint8_t)));

    vm_inputs_map_[vm_input_data_] = std::make_shared<ForthInputBuffer>(ptr, 0, initial_);
    vm_.get()->run(vm_inputs_map_);
  }

  template<typename T>
  void
  LayoutBuilder::set_data(T x) {
    reinterpret_cast<T*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = x;
  }

  void
  LayoutBuilder::resume() const {
    if (vm_.get()->resume() == util::ForthError::user_halt) {
      throw std::invalid_argument(vm_.get()->string_at(vm_.get()->stack().back()));
    }
  }

  void
  LayoutBuilder::debug_step() const {
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
    // FIXME refactoring std::cout << "array:\n" << snapshot().get()->tostring() << "\n";
  }

  const std::string
  LayoutBuilder::vm_source() const {
    return vm_source_;
  }

  const std::shared_ptr<ForthMachine32>
  LayoutBuilder::vm() const {
    if (vm_ != nullptr) {
      return vm_;
    }
    else {
      throw std::invalid_argument(
        std::string("LayoutBuilder is not connected to a Virtual Machine ")
        + FILENAME(__LINE__));
    }
  }

  const std::string
  LayoutBuilder::tostring() const {
    util::TypeStrs typestrs;
    typestrs["char"] = "char";
    typestrs["string"] = "string";
    std::stringstream out;
    out << "<LayoutBuilder length=\"" << length() << "\" type=\""
        << type(typestrs).get()->tostring() << "\"/>";
    return out.str();
  }

  int64_t
  LayoutBuilder::next_id() {
    return LayoutBuilder::next_node_id++;
  }

  int64_t
  LayoutBuilder::next_error_id() {
    return LayoutBuilder::error_id++;
  }

  int64_t
  LayoutBuilder::length() const {
    return length_;
  }

  void
  LayoutBuilder::pre_snapshot() const {
    vm_.get()->maybe_throw(util::ForthError::user_halt, ignore_);
  }

  const TypePtr
  LayoutBuilder::type(const util::TypeStrs& typestrs) const {
    // FIXME refactoring return builder_.get()->snapshot(vm().get()->outputs()).get()->type(typestrs);
    throw std::runtime_error(
      std::string("LayoutBuilder type is obsolete")
      + FILENAME(__LINE__));
  }

  void
  LayoutBuilder::null() {
    vm_.get()->stack_push(static_cast<utype>(state::null));
    resume();
  }

  void
  LayoutBuilder::boolean(bool x) {
    builder_.get()->boolean(x, this);
  }

  template<>
  void
  LayoutBuilder::add<bool>(bool x) {
    set_data<bool>(x);
    vm_.get()->stack_push(static_cast<utype>(state::boolean));
    resume();
  }

  void
  LayoutBuilder::int64(int64_t x) {
    builder_.get()->int64(x, this);
  }

  template<>
  void
  LayoutBuilder::add<int64_t>(int64_t x) {
    set_data<int64_t>(x);
    vm_.get()->stack_push(static_cast<utype>(state::int64));
    resume();
  }

  void
  LayoutBuilder::float64(double x) {
    builder_.get()->float64(x, this);
  }

  template<>
  void
  LayoutBuilder::add<double>(double x) {
    set_data<double>(x);
    vm_.get()->stack_push(static_cast<utype>(state::float64));
    resume();
  }

  void
  LayoutBuilder::complex(std::complex<double> x) {
    builder_.get()->complex(x, this);
  }

  template<>
  void
  LayoutBuilder::add<std::complex<double>>(std::complex<double> x) {
    set_data<std::complex<double>>(x);
    vm_.get()->stack_push(static_cast<utype>(state::complex128));
    resume();
  }

  void
  LayoutBuilder::bytestring(const char* x) {
    throw std::runtime_error(
      std::string("LayoutBuilder a null terminated 'bytestring' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  LayoutBuilder::bytestring(const char* x, int64_t length) {
    for (int64_t i = 0; i < length; i++) {
      set_data<uint8_t>((uint8_t)x[i]);
      vm_.get()->stack_push(static_cast<utype>(state::uint8));
      resume();
    }
  }

  void
  LayoutBuilder::bytestring(const std::string& x) {
    builder_.get()->bytestring(x, this);
  }

  void
  LayoutBuilder::string(const char* x) {
    throw std::runtime_error(
      std::string("LayoutBuilder a null terminated 'string' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  LayoutBuilder::string(const char* x, int64_t length) {
    for (int64_t i = 0; i < length; i++) {
      set_data<uint8_t>((uint8_t)x[i]);
      vm_.get()->stack_push(static_cast<utype>(state::uint8));
      resume();
    }
  }

  void
  LayoutBuilder::string(const std::string& x) {
    builder_.get()->string(x, this);
  }

  template<>
  void
  LayoutBuilder::add<const std::string&>(const std::string& x) {
    begin_list();
    string(x.c_str(), (int64_t)x.length());
    end_list();
  }

  void
  LayoutBuilder::begin_list() {
    builder_.get()->begin_list(this);
  }

  void
  LayoutBuilder::add_begin_list() {
    vm_.get()->stack_push(static_cast<utype>(state::begin_list));
    vm_.get()->resume();
  }

  void
  LayoutBuilder::end_list() {
    builder_.get()->end_list(this);
  }

  void
  LayoutBuilder::add_end_list() {
    vm_.get()->stack_push(static_cast<utype>(state::end_list));
    vm_.get()->resume();
  }

  void
  LayoutBuilder::index(int64_t x) {
    vm_.get()->stack_push((int32_t)x);
    vm_.get()->stack_push(static_cast<utype>(state::index));
    vm_.get()->resume();
  }

  void
  LayoutBuilder::tag(int8_t x) {
    set_data<int8_t>(x);
    vm_.get()->stack_push(static_cast<utype>(state::tag));
    vm_.get()->resume();
  }

}

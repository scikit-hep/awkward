// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
#include "awkward/Content.h"

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/BoolBuilder.h"
#include "awkward/builder/DatetimeBuilder.h"
#include "awkward/builder/Int64Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/StringBuilder.h"
#include "awkward/builder/ListBuilder.h"
#include "awkward/builder/TupleBuilder.h"
#include "awkward/builder/RecordBuilder.h"
#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/UnknownBuilder.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/None.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/Record.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/datetime_util.h"

#include "awkward/io/json.h"

namespace rj = rapidjson;

namespace awkward {

  namespace {

    const ContentPtr
    builder_snapshot(const BuilderPtr builder) {
      ContentPtr out;
      if (builder.get()->classname() == "BoolBuilder") {
        const std::shared_ptr<const BoolBuilder> raw = std::dynamic_pointer_cast<const BoolBuilder>(builder);
        std::vector<ssize_t> shape = { (ssize_t)raw.get()->buffer().length() };
        std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
        out = std::make_shared<NumpyArray>(Identities::none(),
                                               util::Parameters(),
                                               raw.get()->buffer().ptr(),
                                               shape,
                                               strides,
                                               0,
                                               sizeof(bool),
                                               "?",
                                               util::dtype::boolean,
                                               kernel::lib::cpu);

      }
      else if (builder.get()->classname() == "Complex128Builder") {
        const std::shared_ptr<const Complex128Builder> raw = std::dynamic_pointer_cast<const Complex128Builder>(builder);
        std::vector<ssize_t> shape = { (ssize_t)raw.get()->buffer().length() };
        std::vector<ssize_t> strides = { (ssize_t)sizeof(std::complex<double>) };
        out = std::make_shared<NumpyArray>(Identities::none(),
                                               util::Parameters(),
                                               raw.get()->buffer().ptr(),
                                               shape,
                                               strides,
                                               0,
                                               sizeof(std::complex<double>),
                                               "Zd",
                                               util::dtype::complex128,
                                               kernel::lib::cpu);

      }
      else if (builder.get()->classname() == "DatetimeBuilder") {
        const std::shared_ptr<const DatetimeBuilder> raw = std::dynamic_pointer_cast<const DatetimeBuilder>(builder);
        std::vector<ssize_t> shape = { (ssize_t)raw.get()->buffer().length() };
        std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };

        auto dtype = util::name_to_dtype(raw.get()->unit());
        auto format = std::string(util::dtype_to_format(dtype))
          .append(std::to_string(util::dtype_to_itemsize(dtype)))
          .append(util::format_to_units(raw.get()->unit()));
        out = std::make_shared<NumpyArray>(
                 Identities::none(),
                 util::Parameters(),
                 raw.get()->buffer().ptr(),
                 shape,
                 strides,
                 0,
                 sizeof(int64_t),
                 format,
                 dtype,
                 kernel::lib::cpu);

      }
      else if (builder.get()->classname() == "Float64Builder") {
        const std::shared_ptr<const Float64Builder> raw = std::dynamic_pointer_cast<const Float64Builder>(builder);
        std::vector<ssize_t> shape = { (ssize_t)raw.get()->buffer().length() };
        std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
        out = std::make_shared<NumpyArray>(Identities::none(),
                                               util::Parameters(),
                                               raw.get()->buffer().ptr(),
                                               shape,
                                               strides,
                                               0,
                                               sizeof(double),
                                               "d",
                                               util::dtype::float64,
                                               kernel::lib::cpu);
      }
        else if (builder.get()->classname() == "Int64Builder") {
          const std::shared_ptr<const Int64Builder> raw = std::dynamic_pointer_cast<const Int64Builder>(builder);
          std::vector<ssize_t> shape = { (ssize_t)raw.get()->buffer().length() };
          std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };
          return std::make_shared<NumpyArray>(
                   Identities::none(),
                   util::Parameters(),
                   raw.get()->buffer().ptr(),
                   shape,
                   strides,
                   0,
                   sizeof(int64_t),
                   util::dtype_to_format(util::dtype::int64),
                   util::dtype::int64,
                   kernel::lib::cpu);
        }
        else if (builder.get()->classname() == "ListBuilder") {
          const std::shared_ptr<const ListBuilder> raw = std::dynamic_pointer_cast<const ListBuilder>(builder);
          Index64 offsets(raw.get()->buffer().ptr(), 0, raw.get()->buffer().length(), kernel::lib::cpu);
          return std::make_shared<ListOffsetArray64>(Identities::none(),
                                                         util::Parameters(),
                                                         offsets,
                                                         builder_snapshot(raw.get()->builder()));
        }
        else if (builder.get()->classname() == "OptionBuilder") {
          const std::shared_ptr<const OptionBuilder> raw = std::dynamic_pointer_cast<const OptionBuilder>(builder);
          Index64 index(raw.get()->buffer().ptr(), 0, raw.get()->buffer().length(), kernel::lib::cpu);
          return std::make_shared<IndexedOptionArray64>(Identities::none(),
                                          util::Parameters(),
                                          index,
                                          builder_snapshot(raw.get()->builder())).get()->simplify_optiontype();
        }
        else if (builder.get()->classname() == "RecordBuilder") {
          const std::shared_ptr<const RecordBuilder> raw = std::dynamic_pointer_cast<const RecordBuilder>(builder);
          if (raw.get()->length() == -1) {
            return std::make_shared<EmptyArray>(Identities::none(),
                                                    util::Parameters());
          }
          util::Parameters parameters;
          if (raw.get()->nameptr() != nullptr) {
            parameters["__record__"] = util::quote(raw.get()->name());
          }
          ContentPtrVec contents;
          util::RecordLookupPtr recordlookup =
            std::make_shared<util::RecordLookup>();
          for (size_t i = 0;  i < raw.get()->builders().size();  i++) {
            contents.push_back(builder_snapshot(raw.get()->builders()[i]));
            recordlookup.get()->push_back(raw.get()->keys()[i]);
          }
          std::vector<ArrayCachePtr> caches;  // nothing is virtual here
          return std::make_shared<RecordArray>(Identities::none(),
                                               parameters,
                                               contents,
                                               recordlookup,
                                               raw.get()->length(),
                                               caches);
        }
        else if (builder.get()->classname() == "StringBuilder") {
          const std::shared_ptr<const StringBuilder> raw = std::dynamic_pointer_cast<const StringBuilder>(builder);
          util::Parameters char_parameters;
          util::Parameters string_parameters;

          if (raw.get()->encoding() == nullptr) {
            char_parameters["__array__"] = std::string("\"byte\"");
            string_parameters["__array__"] = std::string("\"bytestring\"");
          }
          else if (std::string(raw.get()->encoding()) == std::string("utf-8")) {
            char_parameters["__array__"] = std::string("\"char\"");
            string_parameters["__array__"] = std::string("\"string\"");
          }
          else {
            throw std::invalid_argument(
              std::string("unsupported encoding: ") + util::quote(raw.get()->encoding()));
          }

          Index64 offsets(raw.get()->buffer().ptr(), 0, raw.get()->buffer().length(), kernel::lib::cpu);
          std::vector<ssize_t> shape = { (ssize_t)raw.get()->content().length() };
          std::vector<ssize_t> strides = { (ssize_t)sizeof(uint8_t) };
          ContentPtr content;
          content = std::make_shared<NumpyArray>(Identities::none(),
                                                 char_parameters,
                                                 raw.get()->content().ptr(),
                                                 shape,
                                                 strides,
                                                 0,
                                                 sizeof(uint8_t),
                                                 "B",
                                                 util::dtype::uint8,
                                                 kernel::lib::cpu);
          return std::make_shared<ListOffsetArray64>(Identities::none(),
                                                     string_parameters,
                                                     offsets,
                                                     content);
        }
        else if (builder.get()->classname() == "TupleBuilder") {
          const std::shared_ptr<const TupleBuilder> raw = std::dynamic_pointer_cast<const TupleBuilder>(builder);
          if (raw.get()->length() == -1) {
            return std::make_shared<EmptyArray>(Identities::none(),
                                                    util::Parameters());
          }
          ContentPtrVec contents;
          for (size_t i = 0;  i < raw.get()->contents().size();  i++) {
            contents.push_back(builder_snapshot(raw.get()->contents()[i]));
          }
          std::vector<ArrayCachePtr> caches;  // nothing is virtual here
          return std::make_shared<RecordArray>(Identities::none(),
                                                   util::Parameters(),
                                                   contents,
                                                   util::RecordLookupPtr(nullptr),
                                                   raw.get()->length(),
                                                   caches);
        }
        else if (builder.get()->classname() == "UnionBuilder") {
          const std::shared_ptr<const UnionBuilder> raw = std::dynamic_pointer_cast<const UnionBuilder>(builder);
          Index8 tags(raw.get()->tags().ptr(), 0, raw.get()->tags().length(), kernel::lib::cpu);
          Index64 index(raw.get()->index().ptr(), 0, raw.get()->index().length(), kernel::lib::cpu);
          ContentPtrVec contents;
          for (auto content : raw.get()->contents()) {
            contents.push_back(builder_snapshot(content));
          }
          return std::make_shared<UnionArray8_64>(Identities::none(),
                                    util::Parameters(),
                                    tags,
                                    index,
                                    contents).get()->simplify_uniontype(true, false);

        }
        else if (builder.get()->classname() == "UnknownBuilder") {
          const std::shared_ptr<const UnknownBuilder> raw = std::dynamic_pointer_cast<const UnknownBuilder>(builder);
          if (raw.get()->nullcount() == 0) {
            return std::make_shared<EmptyArray>(Identities::none(),
                                                    util::Parameters());
          }
          else {
            // This is the only snapshot that is O(N), rather than O(1),
            // but it is a corner case (array of only Nones).
            Index64 index(raw.get()->nullcount());
            int64_t* rawptr = index.ptr().get();
            for (int64_t i = 0;  i < raw.get()->nullcount();  i++) {
              rawptr[i] = -1;
            }
            return std::make_shared<IndexedOptionArray64>(
              Identities::none(),
              util::Parameters(),
              index,
              std::make_shared<EmptyArray>(Identities::none(), util::Parameters()));
        }
      }
      return out;
    }
  }
  ////////// writing to JSON
  ToJson::~ToJson() = default;

  void
  ToJson::string(const std::string& x) {
    string(x.c_str(), (int64_t)x.length());
  }

  void
  ToJson::field(const std::string& x) {
    field(x.c_str());
  }

  template <typename DOCUMENT, typename WRITER>
  void copyjson(const DOCUMENT& value, WRITER& writer) {
    if (value.IsNull()) {
      writer.Null();
    }
    else if (value.IsBool()) {
      writer.Bool(value.GetBool());
    }
    else if (value.IsInt()) {
      writer.Int64(value.GetInt());
    }
    else if (value.IsDouble()) {
      writer.Double(value.GetDouble());
    }
    else if (value.IsString()) {
      writer.String(value.GetString());
    }
    else if (value.IsArray()) {
      writer.StartArray();
      for (rapidjson::SizeType i = 0;  i < value.Size();  i++) {
        copyjson(value[i], writer);
      }
      writer.EndArray();
    }
    else if (value.IsObject()) {
      writer.StartObject();
      for (auto it = value.MemberBegin();  it != value.MemberEnd();  ++it) {
        writer.Key(it->name.GetString());
        copyjson(it->value, writer);
      }
      writer.EndObject();
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized JSON element type") + FILENAME(__LINE__));
    }
  }

  class ToJsonString::Impl {
  public:
    Impl(int64_t maxdecimals): buffer_(), writer_(buffer_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }
    void null() { writer_.Null(); }
    void boolean(bool x) { writer_.Bool(x); }
    void integer(int64_t x) { writer_.Int64(x); }
    void real(double x) { writer_.Double(x); }
    void complex(std::complex<double> x,
                 const char* complex_real_string,
                 const char* complex_imag_string) {
      beginrecord();
      field(complex_real_string);
      real(x.real());
      field(complex_imag_string);
      real(x.imag());
      endrecord();
    }
    void string(const char* x, int64_t length) {
      writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
    void json(const char* data) {
      rj::Document doc;
      doc.Parse<rj::kParseNanAndInfFlag>(data);
      copyjson(doc, writer_);
    }
    const std::string tostring() {
      return std::string(buffer_.GetString());
    }
  private:
    rj::StringBuffer buffer_;
    rj::Writer<rj::StringBuffer> writer_;
  };

  ToJsonString::ToJsonString(int64_t maxdecimals,
                             const char* nan_string,
                             const char* infinity_string,
                             const char* minus_infinity_string,
                             const char* complex_real_string,
                             const char* complex_imag_string)
      : impl_(new ToJsonString::Impl(maxdecimals))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string)
      , complex_real_string_(complex_real_string)
      , complex_imag_string_(complex_imag_string) { }

  ToJsonString::~ToJsonString() {
    delete impl_;
  }

  void
  ToJsonString::null() {
    impl_->null();
  }

  void
  ToJsonString::boolean(bool x) {
    impl_->boolean(x);
  }

  void
  ToJsonString::integer(int64_t x) {
    impl_->integer(x);
  }

  void
  ToJsonString::real(double x) {
    if (nan_string_ != nullptr  &&  std::isnan(x)) {
      impl_->string(nan_string_, (int64_t)strlen(nan_string_));
    } else if (infinity_string_ != nullptr  &&  std::isinf(x)  &&  !std::signbit(x)) {
      impl_->string(infinity_string_, (int64_t)strlen(infinity_string_));
    } else if (minus_infinity_string_ != nullptr  &&  std::isinf(x)  &&  std::signbit(x)) {
      impl_->string(minus_infinity_string_, (int64_t)strlen(minus_infinity_string_));
    }
    else {
      impl_->real(x);
    }
  }

  void
  ToJsonString::complex(std::complex<double> x) {
    if (complex_real_string_ != nullptr  &&  complex_imag_string_ != nullptr) {
      impl_->complex(x, complex_real_string_, complex_imag_string_);
    }
    else {
      throw std::invalid_argument(
        std::string("Complex numbers can't be converted to JSON without"
          " setting \'complex_record_fields\' ")
        + FILENAME(__LINE__));
    }
  }

  void
  ToJsonString::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void
  ToJsonString::beginlist() {
    impl_->beginlist();
  }

  void
  ToJsonString::endlist() {
    impl_->endlist();
  }

  void
  ToJsonString::beginrecord() {
    impl_->beginrecord();
  }

  void
  ToJsonString::field(const char* x) {
    impl_->field(x);
  }

  void
  ToJsonString::endrecord() {
    impl_->endrecord();
  }

  void
  ToJsonString::json(const char* x) {
    impl_->json(x);
  }

  const std::string
  ToJsonString::tostring() {
    return impl_->tostring();
  }

  class
  ToJsonPrettyString::Impl {
  public:
    Impl(int64_t maxdecimals): buffer_(), writer_(buffer_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }
    void null() { writer_.Null(); }
    void boolean(bool x) { writer_.Bool(x); }
    void integer(int64_t x) { writer_.Int64(x); }
    void real(double x) { writer_.Double(x); }
    void complex(std::complex<double> x,
                 const char* complex_real_string,
                 const char* complex_imag_string) {
      beginrecord();
      field(complex_real_string);
      real(x.real());
      field(complex_imag_string);
      real(x.imag());
      endrecord();
    }
    void string(const char* x, int64_t length) {
      writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
    void json(const char* data) {
      rj::Document doc;
      doc.Parse<rj::kParseNanAndInfFlag>(data);
      copyjson(doc, writer_);
    }
    const std::string tostring() {
      return std::string(buffer_.GetString());
    }
  private:
    rj::StringBuffer buffer_;
    rj::PrettyWriter<rj::StringBuffer> writer_;
  };

  ToJsonPrettyString::ToJsonPrettyString(int64_t maxdecimals,
                                         const char* nan_string,
                                         const char* infinity_string,
                                         const char* minus_infinity_string,
                                         const char* complex_real_string,
                                         const char* complex_imag_string)
      : impl_(new ToJsonPrettyString::Impl(maxdecimals))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string)
      , complex_real_string_(complex_real_string)
      , complex_imag_string_(complex_imag_string) { }

  ToJsonPrettyString::~ToJsonPrettyString() {
    delete impl_;
  }

  void
  ToJsonPrettyString::null() {
    impl_->null();
  }

  void
  ToJsonPrettyString::boolean(bool x) {
    impl_->boolean(x);
  }

  void
  ToJsonPrettyString::integer(int64_t x) {
    impl_->integer(x);
  }

  void
  ToJsonPrettyString::real(double x) {
    if (nan_string_ != nullptr  &&  std::isnan(x)) {
      impl_->string(nan_string_, (int64_t)strlen(nan_string_));
    } else if (infinity_string_ != nullptr  &&  std::isinf(x)  &&  !std::signbit(x)) {
      impl_->string(infinity_string_, (int64_t)strlen(infinity_string_));
    } else if (minus_infinity_string_ != nullptr  &&  std::isinf(x)  &&  std::signbit(x)) {
      impl_->string(minus_infinity_string_, (int64_t)strlen(minus_infinity_string_));
    }
    else {
      impl_->real(x);
    }
  }

  void
  ToJsonPrettyString::complex(std::complex<double> x) {
    if (complex_real_string_ != nullptr  &&  complex_imag_string_ != nullptr) {
      impl_->complex(x, complex_real_string_, complex_imag_string_);
    }
    else {
      throw std::invalid_argument(
        std::string("Complex numbers can't be converted to JSON without"
          " setting \'complex_record_fields\' ")
        + FILENAME(__LINE__));
    }
  }

  void
  ToJsonPrettyString::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void
  ToJsonPrettyString::beginlist() {
    impl_->beginlist();
  }

  void
  ToJsonPrettyString::endlist() {
    impl_->endlist();
  }

  void
  ToJsonPrettyString::beginrecord() {
    impl_->beginrecord();
  }

  void
  ToJsonPrettyString::field(const char* x) {
    impl_->field(x);
  }

  void
  ToJsonPrettyString::endrecord() {
    impl_->endrecord();
  }

  void
  ToJsonPrettyString::json(const char* x) {
    impl_->json(x);
  }

  const std::string
  ToJsonPrettyString::tostring() {
    return impl_->tostring();
  }

  class ToJsonFile::Impl {
  public:
    Impl(FILE* destination, int64_t maxdecimals, int64_t buffersize)
        : buffer_(kernel::malloc<char>(kernel::lib::cpu, buffersize))
        , stream_(destination,
                  buffer_.get(),
                  ((size_t)buffersize)*sizeof(char))
        , writer_(stream_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }
    void null() { writer_.Null(); }
    void boolean(bool x) { writer_.Bool(x); }
    void integer(int64_t x) { writer_.Int64(x); }
    void real(double x) { writer_.Double(x); }
    void complex(std::complex<double> x,
                 const char* complex_real_string,
                 const char* complex_imag_string) {
      beginrecord();
      field(complex_real_string);
      real(x.real());
      field(complex_imag_string);
      real(x.imag());
      endrecord();
    }
    void string(const char* x, int64_t length) {
      writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
    void json(const char* data) {
      rj::Document doc;
      doc.Parse<rj::kParseNanAndInfFlag>(data);
      copyjson(doc, writer_);
    }
  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream stream_;
    rj::Writer<rj::FileWriteStream> writer_ ;
  };

  ToJsonFile::ToJsonFile(FILE* destination,
                         int64_t maxdecimals,
                         int64_t buffersize,
                         const char* nan_string,
                         const char* infinity_string,
                         const char* minus_infinity_string,
                         const char* complex_real_string,
                         const char* complex_imag_string)
      : impl_(new ToJsonFile::Impl(destination, maxdecimals, buffersize))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string)
      , complex_real_string_(complex_real_string)
      , complex_imag_string_(complex_imag_string) { }

  ToJsonFile::~ToJsonFile() {
    delete impl_;
  }

  void
  ToJsonFile::null() {
    impl_->null();
  }

  void
  ToJsonFile::boolean(bool x) {
    impl_->boolean(x);
  }

  void
  ToJsonFile::integer(int64_t x) {
    impl_->integer(x);
  }

  void
  ToJsonFile::real(double x) {
    if (nan_string_ != nullptr  &&  std::isnan(x)) {
      impl_->string(nan_string_, (int64_t)strlen(nan_string_));
    } else if (infinity_string_ != nullptr  &&  std::isinf(x)  &&  !std::signbit(x)) {
      impl_->string(infinity_string_, (int64_t)strlen(infinity_string_));
    } else if (minus_infinity_string_ != nullptr  &&  std::isinf(x)  &&  std::signbit(x)) {
      impl_->string(minus_infinity_string_, (int64_t)strlen(minus_infinity_string_));
    }
    else {
      impl_->real(x);
    }
  }

  void
  ToJsonFile::complex(std::complex<double> x) {
    if (complex_real_string_ != nullptr  &&  complex_imag_string_ != nullptr) {
      impl_->complex(x, complex_real_string_, complex_imag_string_);
    }
    else {
      throw std::invalid_argument(
        std::string("Complex numbers can't be converted to JSON without"
          " setting \'complex_record_fields\' ")
        + FILENAME(__LINE__));
    }
  }

  void
  ToJsonFile::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void
  ToJsonFile::beginlist() {
    impl_->beginlist();
  }

  void
  ToJsonFile::endlist() {
    impl_->endlist();
  }

  void
  ToJsonFile::beginrecord() {
    impl_->beginrecord();
  }

  void
  ToJsonFile::field(const char* x) {
    impl_->field(x);
  }

  void
  ToJsonFile::endrecord() {
    impl_->endrecord();
  }

  void
  ToJsonFile::json(const char* x) {
    impl_->json(x);
  }

  class ToJsonPrettyFile::Impl {
  public:
    Impl(FILE* destination, int64_t maxdecimals, int64_t buffersize)
        : buffer_(kernel::malloc<char>(kernel::lib::cpu, buffersize))
        , stream_(destination,
                  buffer_.get(),
                  ((size_t)buffersize)*sizeof(char))
        , writer_(stream_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }
    void null() { writer_.Null(); }
    void boolean(bool x) { writer_.Bool(x); }
    void integer(int64_t x) { writer_.Int64(x); }
    void real(double x) { writer_.Double(x); }
    void complex(std::complex<double> x,
                 const char* complex_real_string,
                 const char* complex_imag_string) {
      beginrecord();
      field(complex_real_string);
      real(x.real());
      field(complex_imag_string);
      real(x.imag());
      endrecord();
    }
    void string(const char* x, int64_t length) {
      writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
    void json(const char* data) {
      rj::Document doc;
      doc.Parse<rj::kParseNanAndInfFlag>(data);
      copyjson(doc, writer_);
    }
  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream stream_;
    rj::PrettyWriter<rj::FileWriteStream> writer_;
  };

  ToJsonPrettyFile::ToJsonPrettyFile(FILE* destination,
                                     int64_t maxdecimals,
                                     int64_t buffersize,
                                     const char* nan_string,
                                     const char* infinity_string,
                                     const char* minus_infinity_string,
                                     const char* complex_real_string,
                                     const char* complex_imag_string)
      : impl_(new ToJsonPrettyFile::Impl(destination,
                                         maxdecimals,
                                         buffersize))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string)
      , complex_real_string_(complex_real_string)
      , complex_imag_string_(complex_imag_string) { }

  ToJsonPrettyFile::~ToJsonPrettyFile() {
    delete impl_;
  }

  void
  ToJsonPrettyFile::null() {
    impl_->null();
  }

  void
  ToJsonPrettyFile::boolean(bool x) {
    impl_->boolean(x);
  }

  void
  ToJsonPrettyFile::integer(int64_t x) {
    impl_->integer(x);
  }

  void
  ToJsonPrettyFile::real(double x) {
    if (nan_string_ != nullptr  &&  std::isnan(x)) {
      impl_->string(nan_string_, (int64_t)strlen(nan_string_));
    } else if (infinity_string_ != nullptr  &&  std::isinf(x)  &&  !std::signbit(x)) {
      impl_->string(infinity_string_, (int64_t)strlen(infinity_string_));
    } else if (minus_infinity_string_ != nullptr  &&  std::isinf(x)  &&  std::signbit(x)) {
      impl_->string(minus_infinity_string_, (int64_t)strlen(minus_infinity_string_));
    }
    else {
      impl_->real(x);
    }
  }

  void
  ToJsonPrettyFile::complex(std::complex<double> x) {
    if (complex_real_string_ != nullptr  &&  complex_imag_string_ != nullptr) {
      impl_->complex(x, complex_real_string_, complex_imag_string_);
    }
    else {
      throw std::invalid_argument(
        std::string("Complex numbers can't be converted to JSON without"
          " setting \'complex_record_fields\' ")
        + FILENAME(__LINE__));
    }
  }

  void
  ToJsonPrettyFile::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void
  ToJsonPrettyFile::beginlist() {
    impl_->beginlist();
  }

  void
  ToJsonPrettyFile::endlist() {
    impl_->endlist();
  }

  void
  ToJsonPrettyFile::beginrecord() {
    impl_->beginrecord();
  }

  void
  ToJsonPrettyFile::field(const char* x) {
    impl_->field(x);
  }

  void
  ToJsonPrettyFile::endrecord() {
    impl_->endrecord();
  }

  void
  ToJsonPrettyFile::json(const char* x) {
    impl_->json(x);
  }

  ////////// reading from JSON

  class Handler: public rj::BaseReaderHandler<rj::UTF8<>, Handler> {
  public:
    Handler(const ArrayBuilderOptions& options,
            const char* nan_string,
            const char* infinity_string,
            const char* minus_infinity_string)
        : builder_(options)
        , moved_(false)
        , nan_string_(nan_string)
        , infinity_string_(infinity_string)
        , minus_infinity_string_(minus_infinity_string) { }

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
    String(const char* str, rj::SizeType length, bool copy) {
      moved_ = true;
      if (nan_string_ != nullptr  &&  strcmp(str, nan_string_) == 0) {
        builder_.real(std::numeric_limits<double>::quiet_NaN());
        return true;
      }
      else if (infinity_string_ != nullptr  &&  strcmp(str, infinity_string_) == 0) {
        builder_.real(std::numeric_limits<double>::infinity());
        return true;
      }
      else if (minus_infinity_string_ != nullptr  &&  strcmp(str, minus_infinity_string_) == 0) {
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
    EndArray(rj::SizeType numfields) {
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
    EndObject(rj::SizeType numfields) {
      moved_ = true;
      builder_.endrecord();
      return true;
    }

    bool
    Key(const char* str, rj::SizeType length, bool copy) {
      moved_ = true;
      builder_.field_check(str);
      return true;
    }

    // FIXME: refactor
    const ContentPtr snapshot() const {
      return builder_snapshot(builder_.builder());
    }

  private:
    ArrayBuilder builder_;
    bool moved_;
    const char* nan_string_;
    const char* infinity_string_;
    const char* minus_infinity_string_;
  };

  template<typename HANDLER, typename STREAM>
  const ContentPtr
  do_parse(HANDLER& handler, rj::Reader& reader, STREAM& stream) {
    int64_t number = 0;
    while (stream.Peek() != 0) {
      handler.reset_moved();
      bool fully_parsed = reader.Parse<rj::kParseStopWhenDoneFlag>(stream, handler);
      if (handler.moved()) {
        if (!fully_parsed) {
          if (stream.Peek() == 0) {
            throw std::invalid_argument(
                std::string("incomplete JSON object at the end of the stream")
                + FILENAME(__LINE__));
          }
          else {
            throw std::invalid_argument(
              std::string("JSON File error at char ")
              + std::to_string(stream.Tell()) + std::string(": \'")
              + stream.Peek() + std::string("\'")
              + FILENAME(__LINE__));
          }
        }
        else {
          number++;
        }
      }
      else if (stream.Peek() != 0) {
        throw std::invalid_argument(
          std::string("JSON File error at char ")
          + std::to_string(stream.Tell()) + std::string(": \'")
          + stream.Peek() + std::string("\'")
          + FILENAME(__LINE__));
      }
    }

    ContentPtr obj = handler.snapshot();
    if (number == 1) {
      return obj.get()->getitem_at_nowrap(0);
    }
    else {
      return obj;
    }
  }

  const ContentPtr
  FromJsonString(const char* source,
                 const ArrayBuilderOptions& options,
                 const char* nan_string,
                 const char* infinity_string,
                 const char* minus_infinity_string) {
    rj::Reader reader;
    rj::StringStream stream(source);
    Handler handler(options,
                    nan_string,
                    infinity_string,
                    minus_infinity_string);
    return do_parse(handler, reader, stream);
  }

  //
  const ContentPtr
  FromJsonFile(FILE* source,
               const ArrayBuilderOptions& options,
               int64_t buffersize,
               const char* nan_string,
               const char* infinity_string,
               const char* minus_infinity_string) {
    rj::Reader reader;
    std::shared_ptr<char> buffer = kernel::malloc<char>(kernel::lib::cpu, buffersize);
    rj::FileReadStream stream(source,
                              buffer.get(),
                              ((size_t)buffersize)*sizeof(char));
    Handler handler(options,
                    nan_string,
                    infinity_string,
                    minus_infinity_string);
    return do_parse(handler, reader, stream);
  }
}

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/io/json.cpp", line)

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/error/en.h"

#include "awkward/builder/ArrayBuilder.h"
#include "awkward/array/RegularArray.h"
#include "awkward/Content.h"

#include "awkward/io/json.h"

namespace rj = rapidjson;

namespace awkward {
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
      writer.Int64((int64_t)value.GetDouble());
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
                             const char* minus_infinity_string)
      : impl_(new ToJsonString::Impl(maxdecimals))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string) { }

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
                                         const char* minus_infinity_string)
      : impl_(new ToJsonPrettyString::Impl(maxdecimals))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string) { }

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
        : buffer_(new char[(size_t)buffersize], kernel::array_deleter<char>())
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
                         const char* minus_infinity_string)
      : impl_(new ToJsonFile::Impl(destination, maxdecimals, buffersize))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string) { }

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
        : buffer_(new char[(size_t)buffersize], kernel::array_deleter<char>())
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
                                     const char* minus_infinity_string)
      : impl_(new ToJsonPrettyFile::Impl(destination,
                                         maxdecimals,
                                         buffersize))
      , nan_string_(nan_string)
      , infinity_string_(infinity_string)
      , minus_infinity_string_(minus_infinity_string) { }

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
    Handler(const ArrayBuilderOptions& options)
        : builder_(options)
        , depth_(0)
        , roots_(0) { }

    const ContentPtr snapshot() const {
      if (depth_ == 0) {
        if (roots_ > 1) {
          ContentPtr out = builder_.snapshot();
          return std::make_shared<RegularArray>(Identities::none(),
                                                util::Parameters(),
                                                out,
                                                1);
        }
        else {
          return builder_.snapshot();
        }
      }
      else {
        throw std::invalid_argument(
          std::string("JSON error array or record is not complete ")
          + FILENAME(__LINE__));
      }
    }

    void next_root() {
      roots_++;
    }

    rj::SizeType nroots() {
      return (rj::SizeType)(roots_);
    }

    bool Null()               { builder_.null();              return true; }
    bool Bool(bool x)         { builder_.boolean(x);          return true; }
    bool Int(int x)           { builder_.integer((int64_t)x); return true; }
    bool Uint(unsigned int x) { builder_.integer((int64_t)x); return true; }
    bool Int64(int64_t x)     { builder_.integer(x);          return true; }
    bool Uint64(uint64_t x)   { builder_.integer((int64_t)x); return true; }
    bool Double(double x)     { builder_.real(x);             return true; }

    bool
    String(const char* str, rj::SizeType length, bool copy) {
      builder_.string(str, (int64_t)length);
      return true;
    }

    bool
    StartArray() {
      if (depth_ != 0) {
        builder_.beginlist();
      }
      depth_++;
      return true;
    }

    bool
    EndArray(rj::SizeType numfields) {
      depth_--;
      if (depth_ != 0) {
        builder_.endlist();
      }
      return true;
    }

    bool
    StartObject() {
      if (depth_ == 0) {
        builder_.beginlist();
      }
      depth_++;
      builder_.beginrecord();
      return true;
    }

    bool
    EndObject(rj::SizeType numfields) {
      depth_--;
      builder_.endrecord();
      if (depth_ == 0) {
        builder_.endlist();
      }
      return true;
    }

    bool
    Key(const char* str, rj::SizeType length, bool copy) {
      builder_.field_check(str);
      return true;
    }

  private:
    ArrayBuilder builder_;
    int64_t depth_;
    int64_t roots_;
  };

  class HandlerNanAndInf: public rj::BaseReaderHandler<rj::UTF8<>, HandlerNanAndInf> {
  public:
    HandlerNanAndInf(const ArrayBuilderOptions& options,
      const char* nan_string,
      const char* infinity_string,
      const char* minus_infinity_string,
      bool nan_and_inf_as_float)
        : builder_(options)
        , depth_(0)
        , roots_(0)
        , nan_string_(nan_string)
        , infinity_string_(infinity_string)
        , minus_infinity_string_(minus_infinity_string)
        , nan_and_inf_as_float_(nan_and_inf_as_float) { }

    const ContentPtr snapshot() const {
      if (depth_ == 0) {
        if (roots_ > 1) {
          ContentPtr out = builder_.snapshot();
          return std::make_shared<RegularArray>(Identities::none(),
            util::Parameters(), out, 1);
        }
        else {
          return builder_.snapshot();
        }
      }
      else {
        throw std::invalid_argument(
          std::string("JSON error array or record is not complete ")
          + FILENAME(__LINE__));
      }
    }

    void next_root() {
      roots_++;
    }

    rj::SizeType nroots() {
      return (rj::SizeType)(roots_);
    }

    bool Null()               { builder_.null();              return true; }
    bool Bool(bool x)         { builder_.boolean(x);          return true; }
    bool Int(int x)           { builder_.integer((int64_t)x); return true; }
    bool Uint(unsigned int x) { builder_.integer((int64_t)x); return true; }
    bool Int64(int64_t x)     { builder_.integer(x);          return true; }
    bool Uint64(uint64_t x)   { builder_.integer((int64_t)x); return true; }
    bool Double(double x)     { builder_.real(x);             return true; }

    bool
    String(const char* str, rj::SizeType length, bool copy) {
      if (nan_and_inf_as_float_) {
        if (nan_string_ != nullptr  &&  strcmp(str, nan_string_) == 0) {
          builder_.real(std::numeric_limits<double>::quiet_NaN());
          return true;
        } else if(infinity_string_ != nullptr  &&  strcmp(str, infinity_string_) == 0) {
          builder_.real(std::numeric_limits<double>::infinity());
          return true;
        } else if(minus_infinity_string_ != nullptr  &&  strcmp(str, minus_infinity_string_) == 0) {
          builder_.real(-std::numeric_limits<double>::infinity());
          return true;
        }
      }
      builder_.string(str, (int64_t)length);
      return true;
    }

    bool
    StartArray() {
      if (depth_ != 0) {
        builder_.beginlist();
      }
      depth_++;
      return true;
    }

    bool
    EndArray(rj::SizeType numfields) {
      depth_--;
      if (depth_ != 0) {
        builder_.endlist();
      }
      return true;
    }

    bool
    StartObject() {
      if (depth_ == 0) {
        builder_.beginlist();
      }
      depth_++;
      builder_.beginrecord();
      return true;
    }

    bool
    EndObject(rj::SizeType numfields) {
      depth_--;
      builder_.endrecord();
      if (depth_ == 0) {
        builder_.endlist();
      }
      return true;
    }

    bool
    Key(const char* str, rj::SizeType length, bool copy) {
      builder_.field_check(str);
      return true;
    }

  private:
    ArrayBuilder builder_;
    int64_t depth_;
    int64_t roots_;
    const char* nan_string_;
    const char* infinity_string_;
    const char* minus_infinity_string_;
    bool nan_and_inf_as_float_;
  };

  template<typename HANDLER, typename STREAM>
  const ContentPtr do_parse(HANDLER& handler, rj::Reader& reader, STREAM& stream)
  {
    bool scan = true;
    bool has_error = false;
    bool done = false;

    handler.StartArray();
    while (stream.Peek() != '\0'  &&  scan) {
      scan = false;
      handler.next_root();
      done = reader.Parse<rj::kParseStopWhenDoneFlag>(stream, handler);
      if(std::isspace(stream.Peek())) {
        stream.Take();
        scan = true;
      }
      else if (done  &&  (stream.Peek() == '{'
        ||  stream.Peek() == '['  ||  stream.Peek() == ' '
        ||  stream.Peek() == '"')) {
        scan = true;
      }
      else if (stream.Peek() == '\\') {
        stream.Take();
        if (stream.Peek() == 'n'  ||  'r'  ||  't') {
          stream.Take();
          scan = true;
        } else {
          scan = false;
          has_error = true;
        }
      }
      else if (stream.Peek() == 0) {
        done = true;
        has_error = false;
      }
      else if (stream.Peek() != 0) {
        if (stream.Peek() != '{'  ||  stream.Peek() != '['  ||  stream.Peek() != ' ') {
          has_error = true;
        }
      }
    }
    if (has_error  ||  !done) {
      throw std::invalid_argument(
        std::string("JSON File error at char ")
        + std::to_string(stream.Tell()) + std::string(": \'")
        + stream.Peek() + std::string("\'")
        + FILENAME(__LINE__));
    }
    handler.EndArray(handler.nroots());
    return handler.snapshot();
  }

  const ContentPtr
  FromJsonString(const char* source,
                 const ArrayBuilderOptions& options,
                 const char* nan_string,
                 const char* infinity_string,
                 const char* minus_infinity_string,
                 bool nan_and_inf_as_float) {
    rj::Reader reader;
    rj::StringStream stream(source);
    if(nan_and_inf_as_float  ||  nan_string != nullptr
      ||  nan_string != nullptr  ||  nan_string != nullptr) {
      HandlerNanAndInf handler (options, nan_string, infinity_string,
        minus_infinity_string, nan_and_inf_as_float);
      return do_parse(handler, reader, stream);
    } else {
      Handler handler (options);
      return do_parse(handler, reader, stream);
    }
  }

  const ContentPtr
  FromJsonFile(FILE* source,
               const ArrayBuilderOptions& options,
               int64_t buffersize,
               const char* nan_string,
               const char* infinity_string,
               const char* minus_infinity_string,
               bool nan_and_inf_as_float) {
    rj::Reader reader;
    std::shared_ptr<char> buffer(new char[(size_t)buffersize],
                                kernel::array_deleter<char>());
    rj::FileReadStream stream(source,
                             buffer.get(),
                             ((size_t)buffersize)*sizeof(char));
    if (nan_and_inf_as_float  ||  nan_string != nullptr
     ||  nan_string != nullptr  ||  nan_string != nullptr) {
     HandlerNanAndInf handler(options, nan_string, infinity_string,
       minus_infinity_string, nan_and_inf_as_float);
     return do_parse(handler, reader, stream);
    }
    else {
     Handler handler(options);
     return do_parse(handler, reader, stream);
    }
  }
}

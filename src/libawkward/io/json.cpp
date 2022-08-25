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
#include "awkward/common.h"
#include "awkward/kernel-dispatch.h"

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
    String(const char* str, rj::SizeType length, bool copy) {
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

  private:
    ArrayBuilder& builder_;
    bool moved_;
    const char* nan_string_;
    const char* posinf_string_;
    const char* neginf_string_;
  };

  template<typename HANDLER, typename STREAM>
  const int64_t
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

    return number;
  }

  int64_t
  FromJsonString(const char* source,
                 ArrayBuilder& builder,
                 const char* nan_string,
                 const char* infinity_string,
                 const char* minus_infinity_string) {
    rj::Reader reader;
    rj::StringStream stream(source);
    Handler handler(builder,
                    nan_string,
                    infinity_string,
                    minus_infinity_string);
    return do_parse(handler, reader, stream);
  }

  int64_t
  FromJsonFile(FILE* source,
               ArrayBuilder& builder,
               int64_t buffersize,
               const char* nan_string,
               const char* infinity_string,
               const char* minus_infinity_string) {
    rj::Reader reader;
    std::shared_ptr<char> buffer = kernel::malloc<char>(kernel::lib::cpu, buffersize);
    rj::FileReadStream stream(source,
                              buffer.get(),
                              ((size_t)buffersize)*sizeof(char));
    Handler handler(builder,
                    nan_string,
                    infinity_string,
                    minus_infinity_string);
    return do_parse(handler, reader, stream);
  }

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
      buffer_ = new char[buffersize];
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
      return count_ + static_cast<size_t>(current_ - buffer_);
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

      std::string context = std::string(buffer_, (size_t)stop).substr(start);
      int64_t arrow = current - start;

      size_t pos;

      pos = 0;
      while ((pos = context.find(9, pos)) != std::string::npos) {
        context.replace(pos, 1, "\\t");
        pos++;
        if (pos < arrow) {
          arrow++;
        }
      }

      pos = 0;
      while ((pos = context.find(10, pos)) != std::string::npos) {
        context.replace(pos, 1, "\\n");
        pos++;
        if (pos < arrow) {
          arrow++;
        }
      }

      pos = 0;
      while ((pos = context.find(13, pos)) != std::string::npos) {
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
  #define KeyTableHeader 11         // arg1: number of items
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
      , schema_okay_(true) { }

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
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Uint64(uint64_t x) {
      moved_ = true;
      // std::cout << "uint64 " << x << " " << specializedjson_->debug() << std::endl;

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
          specializedjson_->write_int64(specializedjson_->argument1(), x);
          return true;
        case FillNumber:
          specializedjson_->write_float64(specializedjson_->argument1(), x);
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool Double(double x) {
      moved_ = true;
      // std::cout << "double " << x << " " << specializedjson_->debug() << std::endl;

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
          return schema_okay_ = false;
      }
    }

    bool
    EndObject(rj::SizeType numfields) {
      moved_ = true;
      // std::cout << "endobject " << specializedjson_->debug() << std::endl;

      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          return true;
        case KeyTableHeader:
          return true;
        default:
          return schema_okay_ = false;
      }
    }

    bool
    Key(const char* str, rj::SizeType length, bool copy) {
      moved_ = true;
      // std::cout << "key " << specializedjson_->debug() << std::endl;

      int64_t jump_to;
      specializedjson_->pop_stack();

      // std::cout << "  pop " << specializedjson_->debug() << std::endl;

      switch (specializedjson_->instruction()) {
        case FillIndexedOptionArray:
          specializedjson_->step_forward();
          jump_to = specializedjson_->find_key(str);
          if (jump_to == -1) {
            return schema_okay_ = false;
          }
          else {
            specializedjson_->step_backward();
            specializedjson_->push_stack(jump_to);
            return true;
          }
        case KeyTableHeader:
          jump_to = specializedjson_->find_key(str);
          if (jump_to == -1) {
            return schema_okay_ = false;
          }
          else {
            specializedjson_->push_stack(jump_to);
            return true;
          }
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
        int64_t outi = buffers_uint8_.size();
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
        int64_t outi = buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
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
        if (std::string("uint8") != item[2].GetString()) {
          throw std::invalid_argument(
            "FillBoolean argument 2 (dtype:str) must be 'uint8'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::uint8);
        int64_t outi = buffers_uint8_.size();
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
        int64_t outi = buffers_int64_.size();
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
        int64_t outi = buffers_float64_.size();
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
        int64_t offsetsi = buffers_int64_.size();
        output_which_.push_back(offsetsi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        buffers_int64_[offsetsi].append(0);
        if (std::string("uint8") != item[4].GetString()) {
          throw std::invalid_argument(
            "FillString argument 4 (dtype:str) must be 'uint8'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[3].GetString());
        output_dtypes_.push_back(util::dtype::uint8);
        int64_t contenti = buffers_uint8_.size();
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
        int64_t outi = buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
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


        if (std::string("int64") != item[2].GetString()) {
          throw std::invalid_argument(
            "VarLengthList argument 2 (dtype:str) must be 'int64'" + FILENAME(__LINE__)
          );
        }
        output_names_.push_back(item[1].GetString());
        output_dtypes_.push_back(util::dtype::int64);
        int64_t outi = buffers_int64_.size();
        output_which_.push_back(outi);
        buffers_int64_.push_back(GrowableBuffer<int64_t>(options));
        buffers_int64_[outi].append(0);
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

    string_offsets_.push_back(0);
    for (auto string : strings) {
      string_offsets_.push_back(string_offsets_[string_offsets_.size() - 1] + string.length());
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

}

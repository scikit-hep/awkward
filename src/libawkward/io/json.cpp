// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "rapidjson/reader.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/error/en.h"

#include "awkward/fillable/FillableArray.h"
#include "awkward/Content.h"

#include "awkward/io/json.h"

namespace rj = rapidjson;

namespace awkward {
  /////////////////////////////////////////////////////// writing to JSON

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
    void string(const char* x, int64_t length) { writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
    const std::string tostring() {
      return std::string(buffer_.GetString());
    }
  private:
    rj::StringBuffer buffer_;
    rj::Writer<rj::StringBuffer> writer_;
    // FIXME: rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag
  };

  ToJsonString::ToJsonString(int64_t maxdecimals)
      : impl_(new ToJsonString::Impl(maxdecimals)) { }

  ToJsonString::~ToJsonString() {
    delete impl_;
  }

  void ToJsonString::null() {
    impl_->null();
  }

  void ToJsonString::boolean(bool x) {
    impl_->boolean(x);
  }

  void ToJsonString::integer(int64_t x) {
    impl_->integer(x);
  }

  void ToJsonString::real(double x) {
    impl_->real(x);
  }

  void ToJsonString::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void ToJsonString::beginlist() {
    impl_->beginlist();
  }

  void ToJsonString::endlist() {
    impl_->endlist();
  }

  void ToJsonString::beginrecord() {
    impl_->beginrecord();
  }

  void ToJsonString::field(const char* x) {
    impl_->field(x);
  }

  void ToJsonString::endrecord() {
    impl_->endrecord();
  }

  const std::string ToJsonString::tostring() {
    return impl_->tostring();
  }

  class ToJsonPrettyString::Impl {
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
    void string(const char* x, int64_t length) { writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
    const std::string tostring() {
      return std::string(buffer_.GetString());
    }
  private:
    rj::StringBuffer buffer_;
    rj::PrettyWriter<rj::StringBuffer> writer_;
    // FIXME: rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag
  };

  ToJsonPrettyString::ToJsonPrettyString(int64_t maxdecimals)
      : impl_(new ToJsonPrettyString::Impl(maxdecimals)) { }

  ToJsonPrettyString::~ToJsonPrettyString() {
    delete impl_;
  }

  void ToJsonPrettyString::null() {
    impl_->null();
  }

  void ToJsonPrettyString::boolean(bool x) {
    impl_->boolean(x);
  }

  void ToJsonPrettyString::integer(int64_t x) {
    impl_->integer(x);
  }

  void ToJsonPrettyString::real(double x) {
    impl_->real(x);
  }

  void ToJsonPrettyString::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void ToJsonPrettyString::beginlist() {
    impl_->beginlist();
  }

  void ToJsonPrettyString::endlist() {
    impl_->endlist();
  }

  void ToJsonPrettyString::beginrecord() {
    impl_->beginrecord();
  }

  void ToJsonPrettyString::field(const char* x) {
    impl_->field(x);
  }

  void ToJsonPrettyString::endrecord() {
    impl_->endrecord();
  }

  const std::string ToJsonPrettyString::tostring() {
    return impl_->tostring();
  }

  class ToJsonFile::Impl {
  public:
    Impl(FILE* destination, int64_t maxdecimals, int64_t buffersize)
        : buffer_(new char[(size_t)buffersize], awkward::util::array_deleter<char>())
        , stream_(destination, buffer_.get(), ((size_t)buffersize)*sizeof(char))
        , writer_(stream_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }
    void null() { writer_.Null(); }
    void boolean(bool x) { writer_.Bool(x); }
    void integer(int64_t x) { writer_.Int64(x); }
    void real(double x) { writer_.Double(x); }
    void string(const char* x, int64_t length) { writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream stream_;
    rj::Writer<rj::FileWriteStream> writer_ ;
    // FIXME: rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag
  };

  ToJsonFile::ToJsonFile(FILE* destination, int64_t maxdecimals, int64_t buffersize)
      : impl_(new ToJsonFile::Impl(destination, maxdecimals, buffersize)) { }

  ToJsonFile::~ToJsonFile() {
    delete impl_;
  }

  void ToJsonFile::null() {
    impl_->null();
  }

  void ToJsonFile::boolean(bool x) {
    impl_->boolean(x);
  }

  void ToJsonFile::integer(int64_t x) {
    impl_->integer(x);
  }

  void ToJsonFile::real(double x) {
    impl_->real(x);
  }

  void ToJsonFile::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void ToJsonFile::beginlist() {
    impl_->beginlist();
  }

  void ToJsonFile::endlist() {
    impl_->endlist();
  }

  void ToJsonFile::beginrecord() {
    impl_->beginrecord();
  }

  void ToJsonFile::field(const char* x) {
    impl_->field(x);
  }

  void ToJsonFile::endrecord() {
    impl_->endrecord();
  }

  class ToJsonPrettyFile::Impl {
  public:
    Impl(FILE* destination, int64_t maxdecimals, int64_t buffersize)
        : buffer_(new char[(size_t)buffersize], awkward::util::array_deleter<char>())
        , stream_(destination, buffer_.get(), ((size_t)buffersize)*sizeof(char))
        , writer_(stream_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }
    void null() { writer_.Null(); }
    void boolean(bool x) { writer_.Bool(x); }
    void integer(int64_t x) { writer_.Int64(x); }
    void real(double x) { writer_.Double(x); }
    void string(const char* x, int64_t length) { writer_.String(x, (rj::SizeType)length); }
    void beginlist() { writer_.StartArray(); }
    void endlist() { writer_.EndArray(); }
    void beginrecord() { writer_.StartObject(); }
    void field(const char* x) { writer_.Key(x); }
    void endrecord() { writer_.EndObject(); }
  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream stream_;
    rj::PrettyWriter<rj::FileWriteStream> writer_;
    // FIXME: rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag, rj::UTF8<>, rj::UTF8<>, rj::CrtAllocator<>, rj::kWriteNanAndInfFlag
  };

  ToJsonPrettyFile::ToJsonPrettyFile(FILE* destination, int64_t maxdecimals, int64_t buffersize)
      : impl_(new ToJsonPrettyFile::Impl(destination, maxdecimals, buffersize)) { }

  ToJsonPrettyFile::~ToJsonPrettyFile() {
    delete impl_;
  }

  void ToJsonPrettyFile::null() {
    impl_->null();
  }

  void ToJsonPrettyFile::boolean(bool x) {
    impl_->boolean(x);
  }

  void ToJsonPrettyFile::integer(int64_t x) {
    impl_->integer(x);
  }

  void ToJsonPrettyFile::real(double x) {
    impl_->real(x);
  }

  void ToJsonPrettyFile::string(const char* x, int64_t length) {
    impl_->string(x, length);
  }

  void ToJsonPrettyFile::beginlist() {
    impl_->beginlist();
  }

  void ToJsonPrettyFile::endlist() {
    impl_->endlist();
  }

  void ToJsonPrettyFile::beginrecord() {
    impl_->beginrecord();
  }

  void ToJsonPrettyFile::field(const char* x) {
    impl_->field(x);
  }

  void ToJsonPrettyFile::endrecord() {
    impl_->endrecord();
  }

  /////////////////////////////////////////////////////// reading from JSON

  class Handler: public rj::BaseReaderHandler<rj::UTF8<>, Handler> {
  public:
    Handler(const FillableOptions& options): array_(options), depth_(0) { }

    const std::shared_ptr<Content> snapshot() const {
      return array_.snapshot();
    }

    bool Null()               { array_.null();              return true; }
    bool Bool(bool x)         { array_.boolean(x);          return true; }
    bool Int(int x)           { array_.integer((int64_t)x); return true; }
    bool Uint(unsigned int x) { array_.integer((int64_t)x); return true; }
    bool Int64(int64_t x)     { array_.integer(x);          return true; }
    bool Uint64(uint64_t x)   { array_.integer((int64_t)x); return true; }
    bool Double(double x)     { array_.real(x);             return true; }

    bool String(const char* str, rj::SizeType length, bool copy) {
      array_.string(str, (int64_t)length);
      return true;
    }

    bool StartArray() {
      if (depth_ != 0) {
        array_.beginlist();
      }
      depth_++;
      return true;
    }
    bool EndArray(rj::SizeType numfields) {
      depth_--;
      if (depth_ != 0) {
        array_.endlist();
      }
      return true;
    }

    bool StartObject() {
      if (depth_ == 0) {
        array_.beginlist();
      }
      depth_++;
      array_.beginrecord();
      return true;
    }
    bool EndObject(rj::SizeType numfields) {
      depth_--;
      array_.endrecord();
      if (depth_ == 0) {
        array_.endlist();
      }
      return true;
    }
    bool Key(const char* str, rj::SizeType length, bool copy) {
      array_.field_check(str);
      return true;
    }

  private:
    FillableArray array_;
    int64_t depth_;
  };

  const std::shared_ptr<Content> FromJsonString(const char* source, const FillableOptions& options) {
    Handler handler(options);
    rj::Reader reader;
    rj::StringStream stream(source);
    if (reader.Parse(stream, handler)) {
      return handler.snapshot();
    }
    else {
      throw std::invalid_argument(std::string("JSON error at char ") + std::to_string(reader.GetErrorOffset()) + std::string(": ") + std::string(rj::GetParseError_En(reader.GetParseErrorCode())));
    }
  }

  const std::shared_ptr<Content> FromJsonFile(FILE* source, const FillableOptions& options, int64_t buffersize) {
    Handler handler(options);
    rj::Reader reader;
    std::shared_ptr<char> buffer(new char[(size_t)buffersize], awkward::util::array_deleter<char>());
    rj::FileReadStream stream(source, buffer.get(), ((size_t)buffersize)*sizeof(char));
    if (reader.Parse(stream, handler)) {
      return handler.snapshot();
    }
    else {
      throw std::invalid_argument(std::string("JSON error at char ") + std::to_string(reader.GetErrorOffset()) + std::string(": ") + std::string(rj::GetParseError_En(reader.GetParseErrorCode())));
    }
    return handler.snapshot();
  }
}

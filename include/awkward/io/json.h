// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_JSON_H_
#define AWKWARD_IO_JSON_H_

#include <cstdio>
#include <string>

#include "rapidjson/reader.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/error/en.h"

#include "awkward/fillable/FillableOptions.h"
#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace rj = rapidjson;

namespace awkward {
  class Content;

  const std::shared_ptr<Content> FromJsonString(const char* source, const FillableOptions& options);
  const std::shared_ptr<Content> FromJsonFile(FILE* source, const FillableOptions& options, int64_t buffersize);

  class ToJson {
  public:
    virtual void null() = 0;
    virtual void boolean(bool x) = 0;
    virtual void integer(int64_t x) = 0;
    virtual void real(double x) = 0;
    virtual void beginlist() = 0;
    virtual void endlist() = 0;
    virtual void beginrec() = 0;
    virtual void endrec() = 0;
    virtual void fieldkey(const char* x) = 0;
    virtual void string(const char* x) = 0;
  };

  class ToJsonString: public ToJson {
  public:
    ToJsonString(int64_t maxdecimals): buffer_(), writer_(buffer_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }

    virtual void null() { writer_.Null(); }
    virtual void boolean(bool x) { writer_.Bool(x); }
    virtual void integer(int64_t x) { writer_.Int64(x); }
    virtual void real(double x) { writer_.Double(x); }
    virtual void beginlist() { writer_.StartArray(); }
    virtual void endlist() { writer_.EndArray(); }
    virtual void beginrec() { writer_.StartObject(); }
    virtual void endrec() { writer_.EndObject(); }
    virtual void fieldkey(const char* x) { writer_.Key(x); }
    virtual void string(const char* x) { writer_.String(x); }

    std::string tostring() {
      return std::string(buffer_.GetString());
    }

  private:
    rj::StringBuffer buffer_;
    rj::Writer<rj::StringBuffer> writer_;
  };

  class ToJsonPrettyString: public ToJson {
  public:
    ToJsonPrettyString(int64_t maxdecimals): buffer_(), writer_(buffer_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }

    virtual void null() { writer_.Null(); }
    virtual void boolean(bool x) { writer_.Bool(x); }
    virtual void integer(int64_t x) { writer_.Int64(x); }
    virtual void real(double x) { writer_.Double(x); }
    virtual void beginlist() { writer_.StartArray(); }
    virtual void endlist() { writer_.EndArray(); }
    virtual void beginrec() { writer_.StartObject(); }
    virtual void endrec() { writer_.EndObject(); }
    virtual void fieldkey(const char* x) { writer_.Key(x); }
    virtual void string(const char* x) { writer_.String(x); }

    std::string tostring() {
      return std::string(buffer_.GetString());
    }

  private:
    rj::StringBuffer buffer_;
    rj::PrettyWriter<rj::StringBuffer> writer_;
  };

  class ToJsonFile: public ToJson {
  public:
    ToJsonFile(FILE* destination, int64_t maxdecimals, int64_t buffersize): buffer_(new char[(size_t)buffersize], awkward::util::array_deleter<char>()), stream_(destination, buffer_.get(), ((size_t)buffersize)*sizeof(char)), writer_(stream_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }

    virtual void null() { writer_.Null(); }
    virtual void boolean(bool x) { writer_.Bool(x); }
    virtual void integer(int64_t x) { writer_.Int64(x); }
    virtual void real(double x) { writer_.Double(x); }
    virtual void beginlist() { writer_.StartArray(); }
    virtual void endlist() { writer_.EndArray(); }
    virtual void beginrec() { writer_.StartObject(); }
    virtual void endrec() { writer_.EndObject(); }
    virtual void fieldkey(const char* x) { writer_.Key(x); }
    virtual void string(const char* x) { writer_.String(x); }

  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream stream_;
    rj::Writer<rj::FileWriteStream> writer_;
  };

  class ToJsonPrettyFile: public ToJson {
  public:
    ToJsonPrettyFile(FILE* destination, int64_t maxdecimals, int64_t buffersize): buffer_(new char[(size_t)buffersize], awkward::util::array_deleter<char>()), stream_(destination, buffer_.get(), ((size_t)buffersize)*sizeof(char)), writer_(stream_) {
      if (maxdecimals >= 0) {
        writer_.SetMaxDecimalPlaces((int)maxdecimals);
      }
    }

    virtual void null() { writer_.Null(); }
    virtual void boolean(bool x) { writer_.Bool(x); }
    virtual void integer(int64_t x) { writer_.Int64(x); }
    virtual void real(double x) { writer_.Double(x); }
    virtual void beginlist() { writer_.StartArray(); }
    virtual void endlist() { writer_.EndArray(); }
    virtual void beginrec() { writer_.StartObject(); }
    virtual void endrec() { writer_.EndObject(); }
    virtual void fieldkey(const char* x) { writer_.Key(x); }
    virtual void string(const char* x) { writer_.String(x); }

  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream stream_;
    rj::PrettyWriter<rj::FileWriteStream> writer_;
  };

}

#endif // AWKWARD_IO_JSON_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/fillable/FillableArray.h"
#include "awkward/Content.h"

#include "awkward/io/json.h"

namespace awkward {
  class Handler: public rj::BaseReaderHandler<rj::UTF8<>, Handler> {
  public:
    Handler(const FillableOptions& options): array_(options), depth_(0) { }

    const std::shared_ptr<Content> snapshot() const {
      return array_.snapshot();
    }

    bool Null() { array_.null(); return true; }
    bool Bool(bool x) { array_.boolean(x); return true; }
    bool Int(int x) { array_.integer((int64_t)x); return true; }
    bool Uint(unsigned int x) { array_.integer((int64_t)x); return true; }
    bool Int64(int64_t x) { array_.integer(x); return true; }
    bool Uint64(uint64_t x) { array_.integer((int64_t)x); return true; }
    bool Double(double x) { array_.real(x); return true; }

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
      throw std::runtime_error("not implemented: Handler::StartObject");
    }
    bool EndObject(rj::SizeType numfields) {
      throw std::runtime_error("not implemented: Handler::EndObject");
    }
    bool Key(const char* str, rj::SizeType length, bool copy) {
      throw std::runtime_error("not implemented: Handler::Key");
    }
    bool String(const char* str, rj::SizeType length, bool copy) {
      throw std::runtime_error("not implemented: Handler::String");
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
    rj::FileReadStream stream(source, buffer.get(), buffersize*sizeof(char));
    if (reader.Parse(stream, handler)) {
      return handler.snapshot();
    }
    else {
      throw std::invalid_argument(std::string("JSON error at char ") + std::to_string(reader.GetErrorOffset()) + std::string(": ") + std::string(rj::GetParseError_En(reader.GetParseErrorCode())));
    }
    return handler.snapshot();
  }
}

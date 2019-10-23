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
#include "awkward/Content.h"

namespace rj = rapidjson;

namespace awkward {
  const std::shared_ptr<Content> FromJsonString(const char* source, const FillableOptions& options);
  const std::shared_ptr<Content> FromJsonFile(FILE* source, const FillableOptions& options, int64_t buffersize);

  template <typename W>
  class ToJson {
  public:
    void null();
    void integer(int64_t x);
    void real(double x);
    void string(const char* x);
    void beginlist();
    void endlist();
    void beginrec();
    void fieldname(const char* x);
    void endrec();

  protected:
    std::unique_ptr<W> writer_;
  };

  class ToJsonString: public ToJson<rj::Writer<rj::StringBuffer>> {
    ToJsonString(): buffer_() {
      writer_ = std::unique_ptr<rj::Writer<rj::StringBuffer>>(new rj::Writer<rj::StringBuffer>(buffer_));
    }
    std::string tostring();

  private:
    rj::StringBuffer buffer_;
  };

  class ToJsonPrettyString: public ToJson<rj::PrettyWriter<rj::StringBuffer>> {
    ToJsonPrettyString(): buffer_() {
      writer_ = std::unique_ptr<rj::PrettyWriter<rj::StringBuffer>>(new rj::PrettyWriter<rj::StringBuffer>(buffer_));
    }
    std::string tostring();

  private:
    rj::StringBuffer buffer_;
  };

  class ToJsonFile: public ToJson<rj::FileWriteStream> {
    ToJsonFile(FILE* destination, int64_t buffersize): buffer_(new char[(size_t)buffersize], awkward::util::array_deleter<char>()) {
      writer_ = std::unique_ptr<rj::FileWriteStream>(new rj::FileWriteStream(destination, buffer_.get(), buffersize*sizeof(char)));
    }

  private:
    std::shared_ptr<char> buffer_;
  };

}

#endif // AWKWARD_IO_JSON_H_

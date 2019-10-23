// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_JSON_H_
#define AWKWARD_IO_JSON_H_

#include <cstdio>
#include <string>

#include "rapidjson/reader.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"

#include "awkward/fillable/FillableOptions.h"
#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Content.h"

namespace rj = rapidjson;

namespace awkward {
  const std::shared_ptr<Content> FromJsonString(const char* source, const FillableOptions& options);
  const std::shared_ptr<Content> FromJsonFile(FILE* source, const FillableOptions& options, int64_t buffersize);

  class ToJsonString {
  public:
    ToJsonString(): buffer_(), writer_(buffer_) { }

    void null();
    void integer(int64_t x);
    void real(double x);
    void string(const char* x);
    void beginlist();
    void endlist();
    void beginrec();
    void fieldname(const char* x);
    void endrec();
    const char* tocharstar();
    std::string tostring();

  private:
    rj::StringBuffer buffer_;
    rj::Writer<rj::StringBuffer> writer_;
  };

  class ToJsonFile {
  public:
    ToJsonFile(FILE* destination, int64_t buffersize): buffer_(new char[(size_t)buffersize], awkward::util::array_deleter<char>()), writer_(destination, buffer_.get(), buffersize*sizeof(char)) { }

    void null();
    void integer(int64_t x);
    void real(double x);
    void string(const char* x);
    void beginlist();
    void endlist();
    void beginrec();
    void fieldname(const char* x);
    void endrec();

  private:
    std::shared_ptr<char> buffer_;
    rj::FileWriteStream writer_;
  };
}

#endif // AWKWARD_IO_JSON_H_

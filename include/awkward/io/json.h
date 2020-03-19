// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_JSON_H_
#define AWKWARD_IO_JSON_H_

#include <cstdio>
#include <string>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  class Content;

  EXPORT_SYMBOL ContentPtr FromJsonString(const char* source, const ArrayBuilderOptions& options);
  EXPORT_SYMBOL ContentPtr FromJsonFile(FILE* source, const ArrayBuilderOptions& options, int64_t buffersize);

  class EXPORT_SYMBOL ToJson {
  public:
    virtual void null() = 0;
    virtual void boolean(bool x) = 0;
    virtual void integer(int64_t x) = 0;
    virtual void real(double x) = 0;
    virtual void string(const char* x, int64_t length) = 0;
    virtual void beginlist() = 0;
    virtual void endlist() = 0;
    virtual void beginrecord() = 0;
    virtual void field(const char* x) = 0;
    virtual void endrecord() = 0;
  };

  class EXPORT_SYMBOL ToJsonString: public ToJson {
  public:
    ToJsonString(int64_t maxdecimals);
    ~ToJsonString();
    void null() override;
    void boolean(bool x) override;
    void integer(int64_t x) override;
    void real(double x) override;
    void string(const char* x, int64_t length) override;
    void beginlist() override;
    void endlist() override;
    void beginrecord() override;
    void field(const char* x) override;
    void endrecord() override;
    const std::string tostring();
  private:
    class Impl;
    Impl* impl_;
  };

  class EXPORT_SYMBOL ToJsonPrettyString: public ToJson {
  public:
    ToJsonPrettyString(int64_t maxdecimals);
    ~ToJsonPrettyString();
    void null() override;
    void boolean(bool x) override;
    void integer(int64_t x) override;
    void real(double x) override;
    void string(const char* x, int64_t length) override;
    void beginlist() override;
    void endlist() override;
    void beginrecord() override;
    void field(const char* x) override;
    void endrecord() override;
    const std::string tostring();
  private:
    class Impl;
    Impl* impl_;
  };

  class EXPORT_SYMBOL ToJsonFile: public ToJson {
  public:
    ToJsonFile(FILE* destination, int64_t maxdecimals, int64_t buffersize);
    ~ToJsonFile();
    void null() override;
    void boolean(bool x) override;
    void integer(int64_t x) override;
    void real(double x) override;
    void string(const char* x, int64_t length) override;
    void beginlist() override;
    void endlist() override;
    void beginrecord() override;
    void field(const char* x) override;
    void endrecord() override;
  private:
    class Impl;
    Impl* impl_;
  };

  class EXPORT_SYMBOL ToJsonPrettyFile: public ToJson {
  public:
    ToJsonPrettyFile(FILE* destination, int64_t maxdecimals, int64_t buffersize);
    ~ToJsonPrettyFile();
    void null() override;
    void boolean(bool x) override;
    void integer(int64_t x) override;
    void real(double x) override;
    void string(const char* x, int64_t length) override;
    void beginlist() override;
    void endlist() override;
    void beginrecord() override;
    void field(const char* x) override;
    void endrecord() override;
  private:
    class Impl;
    Impl* impl_;
  };
}

#endif // AWKWARD_IO_JSON_H_

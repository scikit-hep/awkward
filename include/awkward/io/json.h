// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IO_JSON_H_
#define AWKWARD_IO_JSON_H_

#include <cstdio>
#include <string>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/common.h"
#include "awkward/util.h"

namespace awkward {
  class Content;

  /// @brief Convert a JSON-encoded string into a Content array using an
  /// ArrayBuilder.
  ///
  /// @param source Null-terminated string containing any valid JSON data.
  /// @param options Configuration options for building an array with an
  /// ArrayBuilder.
  LIBAWKWARD_EXPORT_SYMBOL const ContentPtr
    FromJsonString(const char* source, const ArrayBuilderOptions& options);

  /// @brief Convert a JSON-encoded file into a Content array using an
  /// ArrayBuilder.
  ///
  /// @param source C file handle to a file containing any valid JSON data.
  /// @param options Configuration options for building an array with an
  /// ArrayBuilder.
  /// @param buffersize Number of bytes for an intermediate buffer.
  LIBAWKWARD_EXPORT_SYMBOL const ContentPtr
    FromJsonFile(FILE* source,
                 const ArrayBuilderOptions& options,
                 int64_t buffersize);

  /// @class ToJson
  ///
  /// Abstract base class for producing JSON data.
  class LIBAWKWARD_EXPORT_SYMBOL ToJson {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~ToJson();

    /// @brief Append a `null` value.
    virtual void
      null() = 0;
    /// @brief Append a boolean value `x`.
    virtual void
      boolean(bool x) = 0;
    /// @brief Append an integer value `x`.
    virtual void
      integer(int64_t x) = 0;
    /// @brief Append a real value `x`.
    virtual void
      real(double x) = 0;
    /// @brief Append a string value `x`.
    virtual void
      string(const char* x, int64_t length) = 0;
    /// @brief Begin a list.
    virtual void
      beginlist() = 0;
    /// @brief End the current list.
    virtual void
      endlist() = 0;
    /// @brief Begin a record.
    virtual void
      beginrecord() = 0;
    /// @brief Insert a key for a key-value pair.
    virtual void
      field(const char* x) = 0;
    /// @brief End the current record.
    virtual void
      endrecord() = 0;
    /// @brief Write raw JSON as a string.
    virtual void
      json(const char* data) = 0;
    /// @brief Append a string value `x`.
    void
      string(const std::string& x);
    /// @brief Insert a key for a key-value pair.
    void
      field(const std::string& x);
  };

  /// @class ToJsonString
  ///
  /// @brief Produces a JSON-formatted string.
  class LIBAWKWARD_EXPORT_SYMBOL ToJsonString: public ToJson {
  public:
    /// @brief Creates a ToJsonString with a full set of parameters.
    ///
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for full precision.
    ToJsonString(int64_t maxdecimals);
    /// @brief Empty destructor; required for some C++ reason.
    ~ToJsonString();
    void
      null() override;
    void
      boolean(bool x) override;
    void
      integer(int64_t x) override;
    void
      real(double x) override;
    void
      string(const char* x, int64_t length) override;
    void
      beginlist() override;
    void
      endlist() override;
    void
      beginrecord() override;
    void
      field(const char* x) override;
    void
      endrecord() override;
    void
      json(const char* data) override;
    /// @brief Return the accumulated data as a string.
    const std::string
      tostring();
  private:
    class Impl;
    Impl* impl_;
  };

  /// @class ToJsonPrettyString
  ///
  /// @brief Produces a pretty JSON-formatted string.
  class LIBAWKWARD_EXPORT_SYMBOL ToJsonPrettyString: public ToJson {
  public:
    /// @brief Creates a ToJsonPrettyString with a full set of parameters.
    ///
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for full precision.
    ToJsonPrettyString(int64_t maxdecimals);
    /// @brief Empty destructor; required for some C++ reason.
    ~ToJsonPrettyString();
    void
      null() override;
    void
      boolean(bool x) override;
    void
      integer(int64_t x) override;
    void
      real(double x) override;
    void
      string(const char* x, int64_t length) override;
    void
      beginlist() override;
    void
      endlist() override;
    void
      beginrecord() override;
    void
      field(const char* x) override;
    void
      endrecord() override;
    void
      json(const char* data) override;
    /// @brief Return the accumulated data as a string.
    const std::string
      tostring();
  private:
    class Impl;
    Impl* impl_;
  };

  /// @class ToJsonFile
  ///
  /// @brief Produces a JSON-formatted file.
  class LIBAWKWARD_EXPORT_SYMBOL ToJsonFile: public ToJson {
  public:
    /// @brief Creates a ToJsonFile with a full set of parameters.
    ///
    /// @param destination C file handle to the file to write.
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for full precision.
    /// @param buffersize Number of bytes for an intermediate buffer.
    ToJsonFile(FILE* destination, int64_t maxdecimals, int64_t buffersize);
    /// @brief Empty destructor; required for some C++ reason.
    ~ToJsonFile();
    void
      null() override;
    void
      boolean(bool x) override;
    void
      integer(int64_t x) override;
    void
      real(double x) override;
    void
      string(const char* x, int64_t length) override;
    void
      beginlist() override;
    void
      endlist() override;
    void
      beginrecord() override;
    void
      field(const char* x) override;
    void
      endrecord() override;
    void
      json(const char* data) override;
  private:
    class Impl;
    Impl* impl_;
  };

  /// @class ToJsonPrettyFile
  ///
  /// @brief Produces a pretty JSON-formatted file.
  class LIBAWKWARD_EXPORT_SYMBOL ToJsonPrettyFile: public ToJson {
  public:
    /// @brief Creates a ToJsonPrettyFile with a full set of parameters.
    ///
    /// @param destination C file handle to the file to write.
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for full precision.
    /// @param buffersize Number of bytes for an intermediate buffer.
    ToJsonPrettyFile(FILE* destination,
                     int64_t maxdecimals,
                     int64_t buffersize);
    /// @brief Empty destructor; required for some C++ reason.
    ~ToJsonPrettyFile();
    void
      null() override;
    void
      boolean(bool x) override;
    void
      integer(int64_t x) override;
    void
      real(double x) override;
    void
      string(const char* x, int64_t length) override;
    void
      beginlist() override;
    void
      endlist() override;
    void
      beginrecord() override;
    void
      field(const char* x) override;
    void
      endrecord() override;
    void
      json(const char* data) override;
  private:
    class Impl;
    Impl* impl_;
  };
}

#endif // AWKWARD_IO_JSON_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYBUILDER_H_
#define AWKWARD_ARRAYBUILDER_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/Builder.h"
#include "awkward/builder/UnknownBuilder.h"

namespace awkward {
  class EXPORT_SYMBOL ArrayBuilder {
  public:
    ArrayBuilder(const ArrayBuilderOptions& options);

    const std::string tostring() const;
    int64_t length() const;
    void clear();
    const std::shared_ptr<Type> type(const std::map<std::string, std::string>& typestrs) const;
    const std::shared_ptr<Content> snapshot() const;
    const std::shared_ptr<Content> getitem_at(int64_t at) const;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    const std::shared_ptr<Content> getitem_field(const std::string& key) const;
    const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const;
    const std::shared_ptr<Content> getitem(const Slice& where) const;

    void null();
    void boolean(bool x);
    void integer(int64_t x);
    void real(double x);
    void bytestring(const char* x);
    void bytestring(const char* x, int64_t length);
    void bytestring(const std::string& x);
    void string(const char* x);
    void string(const char* x, int64_t length);
    void string(const std::string& x);
    void beginlist();
    void endlist();
    void begintuple(int64_t numfields);
    void index(int64_t index);
    void endtuple();
    void beginrecord();
    void beginrecord_fast(const char* name);
    void beginrecord_check(const char* name);
    void beginrecord_check(const std::string& name);
    void field_fast(const char* key);
    void field_check(const char* key);
    void field_check(const std::string& key);
    void endrecord();
    void append(const std::shared_ptr<Content>& array, int64_t at);
    void append_nowrap(const std::shared_ptr<Content>& array, int64_t at);
    void extend(const std::shared_ptr<Content>& array);

  private:
    void maybeupdate(const std::shared_ptr<Builder>& tmp);

    static const char* no_encoding;
    static const char* utf8_encoding;

    std::shared_ptr<Builder> builder_;
  };
}

extern "C" {
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_length(void* arraybuilder, int64_t* result);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_clear(void* arraybuilder);

  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_null(void* arraybuilder);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_boolean(void* arraybuilder, bool x);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_integer(void* arraybuilder, int64_t x);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_real(void* arraybuilder, double x);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_bytestring(void* arraybuilder, const char* x);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_bytestring_length(void* arraybuilder, const char* x, int64_t length);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_string(void* arraybuilder, const char* x);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_string_length(void* arraybuilder, const char* x, int64_t length);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_beginlist(void* arraybuilder);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_endlist(void* arraybuilder);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_begintuple(void* arraybuilder, int64_t numfields);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_index(void* arraybuilder, int64_t index);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_endtuple(void* arraybuilder);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_beginrecord(void* arraybuilder);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_beginrecord_fast(void* arraybuilder, const char* name);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_beginrecord_check(void* arraybuilder, const char* name);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_field_fast(void* arraybuilder, const char* key);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_field_check(void* arraybuilder, const char* key);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_endrecord(void* arraybuilder);
  EXPORT_SYMBOL uint8_t awkward_ArrayBuilder_append_nowrap(void* arraybuilder, const void* shared_ptr_ptr, int64_t at);
}

#endif // AWKWARD_ARRAYBUILDER_H_

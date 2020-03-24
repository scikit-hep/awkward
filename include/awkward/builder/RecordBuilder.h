// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDBUILDER_H_
#define AWKWARD_RECORDBUILDER_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"
#include "awkward/builder/UnknownBuilder.h"

namespace awkward {
  class EXPORT_SYMBOL RecordBuilder: public Builder {
  public:
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options);

    RecordBuilder(const ArrayBuilderOptions& options,
                  const std::vector<BuilderPtr>& contents,
                  const std::vector<std::string>& keys,
                  const std::vector<const char*>& pointers,
                  const std::string& name,
                  const char* nameptr,
                  int64_t length,
                  bool begun,
                  int64_t nextindex,
                  int64_t nexttotry);

    const std::string
      name() const;

    const char*
      nameptr() const;

    /// @brief User-friendly name of this class: `"RecordBuilder"`.
    const std::string
      classname() const override;

    int64_t
      length() const override;

    void
      clear() override;

    const ContentPtr
      snapshot() const override;

    bool
      active() const override;

    const BuilderPtr
      null() override;

    const BuilderPtr
      boolean(bool x) override;

    const BuilderPtr
      integer(int64_t x) override;

    const BuilderPtr
      real(double x) override;

    const BuilderPtr
      string(const char* x, int64_t length, const char* encoding) override;

    const BuilderPtr
      beginlist() override;

    const BuilderPtr
      endlist() override;

    const BuilderPtr
      begintuple(int64_t numfields) override;

    const BuilderPtr
      index(int64_t index) override;

    const BuilderPtr
      endtuple() override;

    const BuilderPtr
      beginrecord(const char* name, bool check) override;

    const BuilderPtr
      field(const char* key, bool check) override;

    const BuilderPtr
      endrecord() override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;

  private:
    const BuilderPtr
      field_fast(const char* key);

    const BuilderPtr
      field_check(const char* key);

    const ArrayBuilderOptions options_;
    std::vector<BuilderPtr> contents_;
    std::vector<std::string> keys_;
    std::vector<const char*> pointers_;
    std::string name_;
    const char* nameptr_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;
    int64_t nexttotry_;

    void
      maybeupdate(int64_t i, const BuilderPtr& tmp);
  };
}

#endif // AWKWARD_RECORDBUILDER_H_

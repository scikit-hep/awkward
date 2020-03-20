// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONBUILDER_H_
#define AWKWARD_UNIONBUILDER_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class TupleBuilder;
  class RecordBuilder;

  class EXPORT_SYMBOL UnionBuilder: public Builder {
  public:
    static const BuilderPtr
      fromsingle(const ArrayBuilderOptions& options,
                 const BuilderPtr& firstcontent);

    UnionBuilder(const ArrayBuilderOptions& options,
                 const GrowableBuffer<int8_t>& types,
                 const GrowableBuffer<int64_t>& offsets,
                 std::vector<BuilderPtr>& contents);

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
    const ArrayBuilderOptions options_;
    GrowableBuffer<int8_t> types_;
    GrowableBuffer<int64_t> offsets_;
    std::vector<BuilderPtr> contents_;
    int8_t current_;
  };
}

#endif // AWKWARD_UNIONBUILDER_H_

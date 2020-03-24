// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNKNOWNBUILDER_H_
#define AWKWARD_UNKNOWNBUILDER_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class EXPORT_SYMBOL UnknownBuilder: public Builder {
  public:
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options);

    UnknownBuilder(const ArrayBuilderOptions& options, int64_t nullcount);

    /// @brief User-friendly name of this class: `"UnknownBuilder"`.
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
    int64_t nullcount_;
  };
}

#endif // AWKWARD_UNKNOWNBUILDER_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_OPTIONFILLABLE_H_
#define AWKWARD_OPTIONFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class OptionFillable: public Fillable {
  public:
    static const std::shared_ptr<Fillable> fromnulls(const FillableOptions& options, int64_t nullcount, std::shared_ptr<Fillable> content);
    static const std::shared_ptr<Fillable> fromvalids(const FillableOptions& options, std::shared_ptr<Fillable> content);

    OptionFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, std::shared_ptr<Fillable> content);

    const std::string classname() const override;
    int64_t length() const override;
    void clear() override;
    const std::shared_ptr<Type> type() const override;
    const std::shared_ptr<Content> snapshot(const std::shared_ptr<Type>& type) const override;

    bool active() const override;
    const std::shared_ptr<Fillable> null() override;
    const std::shared_ptr<Fillable> boolean(bool x) override;
    const std::shared_ptr<Fillable> integer(int64_t x) override;
    const std::shared_ptr<Fillable> real(double x) override;
    const std::shared_ptr<Fillable> string(const char* x, int64_t length, const char* encoding) override;
    const std::shared_ptr<Fillable> beginlist() override;
    const std::shared_ptr<Fillable> endlist() override;
    const std::shared_ptr<Fillable> begintuple(int64_t numfields) override;
    const std::shared_ptr<Fillable> index(int64_t index) override;
    const std::shared_ptr<Fillable> endtuple() override;
    const std::shared_ptr<Fillable> beginrecord(const char* name, bool check) override;
    const std::shared_ptr<Fillable> field(const char* key, bool check) override;
    const std::shared_ptr<Fillable> endrecord() override;

  private:
    const FillableOptions options_;
    GrowableBuffer<int64_t> offsets_;
    std::shared_ptr<Fillable> content_;

    void maybeupdate(const std::shared_ptr<Fillable>& tmp);
  };
}

#endif // AWKWARD_OPTIONFILLABLE_H_

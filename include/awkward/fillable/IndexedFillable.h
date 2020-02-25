// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEXEDFILLABLE_H_
#define AWKWARD_INDEXEDFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class IndexedFillable: public Fillable {
  public:
    static const std::shared_ptr<Fillable> fromnulls(const FillableOptions& options, int64_t nullcount, const std::shared_ptr<Content>& array);

    IndexedFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<Fillable>& content, bool hasnull);

    const std::string classname() const override;
    int64_t length() const override;
    void clear() override;
    const std::shared_ptr<Content> snapshot() const override;

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
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;

  private:
    const FillableOptions options_;
    GrowableBuffer<int64_t> index_;
    std::shared_ptr<Fillable> content_;
    bool hasnull_;
  };
}

#endif // AWKWARD_INDEXEDFILLABLE_H_

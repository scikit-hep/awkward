// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_TUPLEFILLABLE_H_
#define AWKWARD_TUPLEFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class TupleFillable: public Fillable {
  public:
    TupleFillable(const FillableOptions& options, const std::vector<std::shared_ptr<Fillable>>& contents, int64_t length, bool begun, size_t nextindex)
        : options_(options)
        , contents_(contents)
        , length_(length)
        , begun_(begun)
        , nextindex_(nextindex) { }

    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options);

    const std::string classname() const override { return "TupleFillable"; };
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

    int64_t numfields() const { return (int64_t)contents_.size(); }

  private:
    const FillableOptions options_;
    std::vector<std::shared_ptr<Fillable>> contents_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;

    void maybeupdate(int64_t i, const std::shared_ptr<Fillable>& tmp);
  };
}

#endif // AWKWARD_TUPLEFILLABLE_H_

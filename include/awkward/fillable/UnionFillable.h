// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONFILLABLE_H_
#define AWKWARD_UNIONFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class TupleFillable;
  class RecordFillable;

  class UnionFillable: public Fillable {
  public:
    UnionFillable(const FillableOptions& options, const GrowableBuffer<int8_t>& types, const GrowableBuffer<int64_t>& offsets, std::vector<std::shared_ptr<Fillable>> contents): options_(options), types_(types), offsets_(offsets), contents_(contents), current_(-1) { }

    static const std::shared_ptr<Fillable> fromsingle(const FillableOptions& options, const std::shared_ptr<Fillable> firstcontent);

    virtual const std::string classname() const { return "UnionFillable"; };
    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual const std::shared_ptr<Fillable> null();
    virtual const std::shared_ptr<Fillable> boolean(bool x);
    virtual const std::shared_ptr<Fillable> integer(int64_t x);
    virtual const std::shared_ptr<Fillable> real(double x);
    virtual const std::shared_ptr<Fillable> beginlist();
    virtual const std::shared_ptr<Fillable> endlist();
    virtual const std::shared_ptr<Fillable> begintuple(int64_t numfields);
    virtual const std::shared_ptr<Fillable> index(int64_t index);
    virtual const std::shared_ptr<Fillable> endtuple();
    virtual const std::shared_ptr<Fillable> beginrecord(const char* name, bool check);
    virtual const std::shared_ptr<Fillable> field(const char* key, bool check);
    virtual const std::shared_ptr<Fillable> endrecord();

  private:
    const FillableOptions options_;
    GrowableBuffer<int8_t> types_;
    GrowableBuffer<int64_t> offsets_;
    std::vector<std::shared_ptr<Fillable>> contents_;
    int8_t current_;
  };
}

#endif // AWKWARD_UNIONFILLABLE_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INT64FILLABLE_H_
#define AWKWARD_INT64FILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class Int64Fillable: public Fillable {
  public:
    Int64Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& buffer): options_(options), buffer_(buffer) { }

    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options);

    virtual const std::string classname() const { return "Int64Fillable"; };
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

    const GrowableBuffer<int64_t> buffer() const { return buffer_; }

  private:
    const FillableOptions options_;
    GrowableBuffer<int64_t> buffer_;
  };
}

#endif // AWKWARD_INT64FILLABLE_H_

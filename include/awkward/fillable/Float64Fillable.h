// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FLOAT64FILLABLE_H_
#define AWKWARD_FLOAT64FILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class Float64Fillable: public Fillable {
  public:
    Float64Fillable(const FillableOptions& options, const GrowableBuffer<double>& buffer): options_(options), buffer_(buffer) { }

    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options);
    static const std::shared_ptr<Fillable> fromint64(const FillableOptions& options, GrowableBuffer<int64_t> old);

    virtual const std::string classname() const { return "Float64Fillable"; };
    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot(const std::shared_ptr<Type> type) const;

    virtual bool active() const;
    virtual const std::shared_ptr<Fillable> null();
    virtual const std::shared_ptr<Fillable> boolean(bool x);
    virtual const std::shared_ptr<Fillable> integer(int64_t x);
    virtual const std::shared_ptr<Fillable> real(double x);
    virtual const std::shared_ptr<Fillable> string(const char* x, int64_t length, const char* encoding);
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
    GrowableBuffer<double> buffer_;
  };
}

#endif // AWKWARD_FLOAT64FILLABLE_H_

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
    Float64Fillable(const FillableOptions& options): options_(options), buffer_(options) { }

    static Float64Fillable* fromsingle(const FillableOptions& options, std::shared_ptr<int64_t> data);

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual Fillable* null();
    virtual Fillable* boolean(bool x);
    virtual Fillable* integer(int64_t x);
    virtual Fillable* real(double x);

  private:
    const FillableOptions options_;
    GrowableBuffer<double> buffer_;
  };
}

#endif // AWKWARD_FLOAT64FILLABLE_H_

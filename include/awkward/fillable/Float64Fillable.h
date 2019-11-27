// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FLOAT64FILLABLE_H_
#define AWKWARD_FLOAT64FILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class FillableArray;

  class Float64Fillable: public Fillable {
  public:
    Float64Fillable(FillableArray* fillablearray, const FillableOptions& options): fillablearray_(fillablearray), options_(options), buffer_(options) { }
    Float64Fillable(FillableArray* fillablearray, const FillableOptions& options, const GrowableBuffer<double>& buffer): fillablearray_(fillablearray), options_(options), buffer_(buffer) { }

    static Float64Fillable* fromint64(FillableArray* fillablearray, const FillableOptions& options, GrowableBuffer<int64_t> old) {
      GrowableBuffer<double> buffer = GrowableBuffer<double>::empty(options, old.reserved());
      int64_t* oldraw = old.ptr().get();
      double* newraw = buffer.ptr().get();
      for (int64_t i = 0;  i < old.length();  i++) {
        newraw[i] = (double)oldraw[i];
      }
      buffer.set_length(old.length());
      return new Float64Fillable(fillablearray, options, buffer);
    }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual Fillable* null();
    virtual Fillable* boolean(bool x);
    virtual Fillable* integer(int64_t x);
    virtual Fillable* real(double x);
    virtual Fillable* beginlist();
    virtual Fillable* endlist();
    virtual Fillable* begintuple(int64_t numfields);
    virtual Fillable* index(int64_t index);
    virtual Fillable* endtuple();
    virtual Fillable* beginrecord(int64_t disambiguator);
    virtual Fillable* field_fast(const char* key);
    virtual Fillable* field_check(const char* key);
    virtual Fillable* endrecord();

  private:
    FillableArray* fillablearray_;
    const FillableOptions options_;
    GrowableBuffer<double> buffer_;
  };
}

#endif // AWKWARD_FLOAT64FILLABLE_H_

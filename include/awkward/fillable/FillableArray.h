// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLEARRAY_H_
#define AWKWARD_FILLABLEARRAY_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class FillableArray {
  public:
    FillableArray(const FillableOptions& options): fillable_(new UnknownFillable(options)) { }

    const std::string tostring() const;
    int64_t length() const;
    void clear();
    const std::shared_ptr<Type> type() const;
    const std::shared_ptr<Content> snapshot() const;
    const std::shared_ptr<Content> getitem_at(int64_t at) const;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    const std::shared_ptr<Content> getitem(const Slice& where) const;

    void null();
    void boolean(bool x);
    void integer(int64_t x);
    void real(double x);
    void beginlist();
    void endlist();

    template <typename T>
    void fill(const std::vector<T>& vector) {
      beginlist();
      for (auto x : vector) {
        fill(x);
      }
      endlist();
    }
    void fill(int64_t x) { integer(x); }
    void fill(double x) { real(x); }

  private:
    std::shared_ptr<Fillable> fillable_;

    void maybeupdate(Fillable* tmp);
  };
}

extern "C" {
  bool awkward_FillableArray_length(void* fillablearray, int64_t* result);
  bool awkward_FillableArray_clear(void* fillablearray);

  bool awkward_FillableArray_null(void* fillablearray);
  bool awkward_FillableArray_boolean(void* fillablearray, bool x);
  bool awkward_FillableArray_integer(void* fillablearray, int64_t x);
  bool awkward_FillableArray_real(void* fillablearray, double x);
  bool awkward_FillableArray_beginlist(void* fillablearray);
  bool awkward_FillableArray_endlist(void* fillablearray);
}

#endif // AWKWARD_FILLABLE_H_

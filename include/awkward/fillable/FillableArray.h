// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLEARRAY_H_
#define AWKWARD_FILLABLEARRAY_H_

#include <unordered_map>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class FillableArray {
  public:
    FillableArray(const FillableOptions& options): fillable_(new UnknownFillable(options)), slots_() { }

    const std::string tostring() const;
    int64_t length() const;
    void clear();
    const std::shared_ptr<Type> type() const;
    const std::shared_ptr<Content> snapshot() const;
    const std::shared_ptr<Content> getitem_at(int64_t at) const;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    const std::shared_ptr<Content> getitem_field(const std::string& key) const;
    const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const;
    const std::shared_ptr<Content> getitem(const Slice& where) const;

    void add_slots(int64_t key, const Slots& slots);

    void null();
    void boolean(bool x);
    void integer(int64_t x);
    void real(double x);
    void beginlist();
    void endlist();
    void beginrec(const Slots& slots);
    void beginrec(int64_t key);
    void reckey(int64_t index);
    void endrec();

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
    std::unordered_map<int64_t, Slots> slots_;

    void maybeupdate(Fillable* tmp);
  };
}

extern "C" {
  uint8_t awkward_FillableArray_length(void* fillablearray, int64_t* result);
  uint8_t awkward_FillableArray_clear(void* fillablearray);

  uint8_t awkward_FillableArray_null(void* fillablearray);
  uint8_t awkward_FillableArray_boolean(void* fillablearray, bool x);
  uint8_t awkward_FillableArray_integer(void* fillablearray, int64_t x);
  uint8_t awkward_FillableArray_real(void* fillablearray, double x);
  uint8_t awkward_FillableArray_beginlist(void* fillablearray);
  uint8_t awkward_FillableArray_endlist(void* fillablearray);
}

#endif // AWKWARD_FILLABLE_H_

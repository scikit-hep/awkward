// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONFILLABLE_H_
#define AWKWARD_UNIONFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class UnionFillable: public Fillable {
  public:
    UnionFillable(const FillableOptions& options, const GrowableBuffer<int8_t>& types, const GrowableBuffer<int64_t>& offsets, std::vector<std::shared_ptr<Fillable>> contents): options_(options), types_(types), offsets_(offsets), contents_(contents) { }

    static UnionFillable* fromsingle(const FillableOptions& options, Fillable* firstcontent) {
      GrowableBuffer<int8_t> types = GrowableBuffer<int8_t>::full(options, 0, firstcontent->length());
      GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::arange(options, firstcontent->length());
      std::vector<std::shared_ptr<Fillable>> contents({ std::shared_ptr<Fillable>(firstcontent) });
      return new UnionFillable(options, types, offsets, contents);
    }

    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual Fillable* null();
    virtual Fillable* boolean(bool x);
    virtual Fillable* integer(int64_t x);
    virtual Fillable* real(double x);
    virtual Fillable* beginlist();
    virtual Fillable* endlist();
    virtual Fillable* beginrec(const std::vector<std::string>& keys);
    virtual Fillable* reckey(int64_t index);
    virtual Fillable* endrec();

  private:
    const FillableOptions options_;
    GrowableBuffer<int8_t> types_;
    GrowableBuffer<int64_t> offsets_;
    std::vector<std::shared_ptr<Fillable>> contents_;

    template <typename T>
    T* findfillable(int8_t& type);
    template <typename T>
    T* maybenew(T* fillable, int64_t& length);
    template <typename T1>
    Fillable* get1(int8_t& type, int64_t& length);
    template <typename T1, typename T2>
    Fillable* get2(int8_t& type, int64_t& length);
  };
}

#endif // AWKWARD_UNIONFILLABLE_H_

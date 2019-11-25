// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <string>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class Slots {
  public:
    Slots(const std::vector<std::string>& slots): slots_(slots) { }

    int64_t numslots() const { return (int64_t)slots_.size(); }
    const std::vector<std::string> slots() const { return slots_; }
    const std::string slot(int64_t index) const { return slots_[(size_t)index]; }

  private:
    const std::vector<std::string> slots_;
  };

  class Fillable {
  public:
    virtual ~Fillable() { }

    virtual int64_t length() const = 0;
    virtual void clear() = 0;
    virtual const std::shared_ptr<Type> type() const = 0;
    virtual const std::shared_ptr<Content> snapshot() const = 0;

    virtual Fillable* null() = 0;
    virtual Fillable* boolean(bool x) = 0;
    virtual Fillable* integer(int64_t x) = 0;
    virtual Fillable* real(double x) = 0;
    virtual Fillable* beginlist() = 0;
    virtual Fillable* endlist() = 0;
    virtual Fillable* beginrec(int64_t slotsid) = 0;
    virtual Fillable* indexrec(int64_t index) = 0;
    virtual Fillable* endrec() = 0;
  };
}

#endif // AWKWARD_FILLABLE_H_

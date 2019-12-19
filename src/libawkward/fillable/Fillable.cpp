// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/fillable/Fillable.h"

namespace awkward {
  Fillable::~Fillable() { }

  void Fillable::setthat(const std::shared_ptr<Fillable>& that) {
    that_ = that;
  }
}

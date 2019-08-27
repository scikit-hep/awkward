// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITY_H_
#define AWKWARD_IDENTITY_H_

#include "awkward/util.h"

namespace awkward {
  class Identity {
  public:
    Identity(const RefType ref, const Index keys, const IndexType chunkdepth, const IndexType keydepth)
        : ref_(ref)
        , keys_(keys)
        , chunkdepth_(chunkdepth)
        , keydepth_(keydepth) { }

    const RefType ref() const { return ref_; }
    const Index keys() const { return keys_; }
    const IndexType chunkdepth() const { return chunkdepth_; }
    const IndexType keydepth() const { return keydepth; }

  private:
    const Index keys_;
    const RefType ref_;
    const IndexType chunkdepth_;
    const IndexType keydepth_;
  };
}

#endif // AWKWARD_IDENTITY_H_

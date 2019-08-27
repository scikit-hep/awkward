// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITY_H_
#define AWKWARD_IDENTITY_H_

#include <atomic>
#include <iomanip>
#include <utility>
#include <string>
#include <vector>

#include "awkward/util.h"
#include "awkward/Index.h"

namespace awkward {
  typedef std::vector<std::pair<IndexType, std::string>> FieldLocation;

  class Identity {
  public:
    static RefType newref();

    Identity(const Index keys, const FieldLocation fieldloc, const IndexType chunkdepth, const IndexType indexdepth, const RefType ref)
        : keys_(keys)
        , fieldloc_(fieldloc)
        , chunkdepth_(chunkdepth)
        , indexdepth_(indexdepth)
        , ref_(ref) { }

    const Index keys() const { return keys_; }
    const FieldLocation fieldloc() const { return fieldloc_; }
    const IndexType chunkdepth() const { return chunkdepth_; }
    const IndexType indexdepth() const { return indexdepth_; }
    const RefType ref() const { return ref_; }

    const IndexType keydepth() const { return (sizeof(ChunkOffsetType)/sizeof(IndexType))*chunkdepth_ + indexdepth_; }

    const std::string repr(const std::string indent, const std::string pre, const std::string post) const;

  private:
    const Index keys_;
    const FieldLocation fieldloc_;
    const IndexType chunkdepth_;
    const IndexType indexdepth_;
    const RefType ref_;
  };
}

#endif // AWKWARD_IDENTITY_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITY_H_
#define AWKWARD_IDENTITY_H_

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
    Identity(const RefType ref, const FieldLocation fieldloc, const Index keys, const IndexType chunkdepth, const IndexType indexdepth)
        : ref_(ref)
        , fieldloc_(fieldloc)
        , keys_(keys)
        , chunkdepth_(chunkdepth)
        , indexdepth_(indexdepth) { }

    const RefType ref() const { return ref_; }
    const FieldLocation fieldloc() const { return fieldloc_; }
    const Index keys() const { return keys_; }
    const IndexType chunkdepth() const { return chunkdepth_; }
    const IndexType indexdepth() const { return indexdepth_; }

    const IndexType keydepth() const { return (sizeof(ChunkOffsetType)/sizeof(IndexType))*chunkdepth_ + indexdepth_; }

    const std::string repr(const std::string indent, const std::string pre, const std::string post) const;

  private:
    const RefType ref_;
    const FieldLocation fieldloc_;
    const Index keys_;
    const IndexType chunkdepth_;
    const IndexType indexdepth_;
  };
}

#endif // AWKWARD_IDENTITY_H_

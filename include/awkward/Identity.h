// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITY_H_
#define AWKWARD_IDENTITY_H_

#include <atomic>
#include <iomanip>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <sstream>

#include "awkward/util.h"

namespace awkward {
  typedef std::vector<std::pair<IndexType, std::string>> FieldLocation;

  class Identity {
  public:
    static RefType newref();
    static IndexType keydepth(IndexType chunkdepth, IndexType indexdepth);

    Identity(const RefType ref, const FieldLocation fieldloc, IndexType chunkdepth, IndexType indexdepth, IndexType length)
        : ref_(ref)
        , fieldloc_(fieldloc)
        , chunkdepth_(chunkdepth)
        , indexdepth_(indexdepth)
        , ptr_(std::shared_ptr<IndexType>(new IndexType[length*Identity::keydepth(chunkdepth, indexdepth)]))
        , offset_(0)
        , length_(length) { }
    Identity(const RefType ref, const FieldLocation fieldloc, IndexType chunkdepth, IndexType indexdepth, const std::shared_ptr<IndexType> ptr, IndexType offset, IndexType length)
        : ref_(ref)
        , fieldloc_(fieldloc)
        , chunkdepth_(chunkdepth)
        , indexdepth_(indexdepth)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    const RefType ref() const { return ref_; }
    const FieldLocation fieldloc() const { return fieldloc_; }
    const IndexType chunkdepth() const { return chunkdepth_; }
    const IndexType indexdepth() const { return indexdepth_; }
    const std::shared_ptr<IndexType> ptr() const { return ptr_; }
    const IndexType offset() const { return offset_; }
    const IndexType length() const { return length_; }

    const IndexType keydepth() const { return keydepth(chunkdepth_, indexdepth_); }

    const std::string repr(const std::string indent, const std::string pre, const std::string post) const;

  private:
    const RefType ref_;
    const FieldLocation fieldloc_;
    const IndexType chunkdepth_;
    const IndexType indexdepth_;
    const std::shared_ptr<IndexType> ptr_;
    IndexType offset_;
    IndexType length_;
  };
}

#endif // AWKWARD_IDENTITY_H_

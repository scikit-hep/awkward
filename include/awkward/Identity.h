// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITY_H_
#define AWKWARD_IDENTITY_H_

#include <cassert>
#include <atomic>
#include <iomanip>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  typedef std::vector<std::pair<int64_t, std::string>> FieldLoc;

  static Ref newref();

  class Identity {
  public:
    Identity(const Ref ref, const FieldLoc fieldloc, int64_t offset, int64_t width, int64_t length)
        : ref_(ref)
        , fieldloc_(fieldloc)
        , offset_(offset)
        , width_(width)
        , length_(length) { }

    const Ref ref() const { return ref_; }
    const FieldLoc fieldloc() const { return fieldloc_; }
    const int64_t offset() const { return offset_; }
    const int64_t width() const { return width_; }
    const int64_t length() const { return length_; }

    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const = 0;
    virtual const std::shared_ptr<Identity> slice(int64_t start, int64_t stop) const = 0;

  private:
    const Ref ref_;
    const FieldLoc fieldloc_;
    int64_t offset_;
    int64_t width_;
    int64_t length_;
  };

  template <typename T>
  class IdentityOf: public Identity {
  public:
    IdentityOf<T>(const Ref ref, const FieldLoc fieldloc, int64_t width, int64_t length)
        : Identity(ref, fieldloc, 0, width, length)
        , ptr_(std::shared_ptr<T>(new T[length*width])) { }
    IdentityOf<T>(const Ref ref, const FieldLoc fieldloc, int64_t offset, int64_t width, int64_t length, const std::shared_ptr<T> ptr)
        : Identity(ref, fieldloc, offset, width, length)
        , ptr_(ptr) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }

    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    virtual const std::shared_ptr<Identity> slice(int64_t start, int64_t stop) const;

  private:
    const std::shared_ptr<T> ptr_;
  };

  typedef IdentityOf<int32_t> Identity32;
  typedef IdentityOf<int64_t> Identity64;
}

#endif // AWKWARD_IDENTITY_H_

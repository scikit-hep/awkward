// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IDENTITY_H_
#define AWKWARD_IDENTITY_H_

#include <string>
#include <vector>
#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Index.h"

namespace awkward {
  class Identity {
  public:
    typedef int64_t Ref;
    typedef std::vector<std::pair<int64_t, std::string>> FieldLoc;

    static Ref newref();
    static std::shared_ptr<Identity> none();

    Identity(const Ref ref, const FieldLoc& fieldloc, int64_t offset, int64_t width, int64_t length);
    const Ref ref() const;
    const FieldLoc fieldloc() const;
    const int64_t offset() const;
    const int64_t width() const;
    const int64_t length() const;

    virtual const std::string classname() const = 0;
    virtual const std::string location_at(int64_t where) const = 0;
    virtual const std::shared_ptr<Identity> to64() const = 0;
    virtual const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const = 0;
    virtual const std::shared_ptr<Identity> getitem_range_nowrap(int64_t start, int64_t stop) const = 0;
    virtual const std::shared_ptr<Identity> shallow_copy() const = 0;
    virtual const std::shared_ptr<Identity> getitem_carry_64(const Index64& carry) const = 0;
    virtual const std::shared_ptr<Identity> withfieldloc(const FieldLoc& fieldloc) const = 0;
    virtual int64_t value(int64_t row, int64_t col) const = 0;

    const std::string tostring() const;

  protected:
    const Ref ref_;
    const FieldLoc fieldloc_;
    int64_t offset_;
    int64_t width_;
    int64_t length_;
  };

  template <typename T>
  class IdentityOf: public Identity {
  public:
    IdentityOf<T>(const Ref ref, const FieldLoc& fieldloc, int64_t width, int64_t length);
    IdentityOf<T>(const Ref ref, const FieldLoc& fieldloc, int64_t offset, int64_t width, int64_t length, const std::shared_ptr<T> ptr);

    const std::shared_ptr<T> ptr() const;

    const std::string classname() const override;
    const std::string location_at(int64_t at) const override;
    const std::shared_ptr<Identity> to64() const override;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Identity> getitem_range_nowrap(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Identity> shallow_copy() const override;
    const std::shared_ptr<Identity> getitem_carry_64(const Index64& carry) const override;
    const std::shared_ptr<Identity> withfieldloc(const FieldLoc& fieldloc) const override;
    int64_t value(int64_t row, int64_t col) const override;

    const std::vector<T> getitem_at(int64_t at) const;
    const std::vector<T> getitem_at_nowrap(int64_t at) const;
    const std::shared_ptr<Identity> getitem_range(int64_t start, int64_t stop) const;

  private:
    const std::shared_ptr<T> ptr_;
  };

  typedef IdentityOf<int32_t> Identity32;
  typedef IdentityOf<int64_t> Identity64;
}

#endif // AWKWARD_IDENTITY_H_

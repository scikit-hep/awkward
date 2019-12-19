// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <string>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class Fillable {
  public:
    virtual ~Fillable();

    virtual const std::string classname() const = 0;
    virtual int64_t length() const = 0;
    virtual void clear() = 0;
    virtual const std::shared_ptr<Type> type() const = 0;
    virtual const std::shared_ptr<Content> snapshot(const std::shared_ptr<Type>& type) const = 0;

    virtual bool active() const = 0;
    virtual const std::shared_ptr<Fillable> null() = 0;
    virtual const std::shared_ptr<Fillable> boolean(bool x) = 0;
    virtual const std::shared_ptr<Fillable> integer(int64_t x) = 0;
    virtual const std::shared_ptr<Fillable> real(double x) = 0;
    virtual const std::shared_ptr<Fillable> string(const char* x, int64_t length, const char* encoding) = 0;
    virtual const std::shared_ptr<Fillable> beginlist() = 0;
    virtual const std::shared_ptr<Fillable> endlist() = 0;
    virtual const std::shared_ptr<Fillable> begintuple(int64_t numfields) = 0;
    virtual const std::shared_ptr<Fillable> index(int64_t index) = 0;
    virtual const std::shared_ptr<Fillable> endtuple() = 0;
    virtual const std::shared_ptr<Fillable> beginrecord(const char* name, bool check) = 0;
    virtual const std::shared_ptr<Fillable> field(const char* key, bool check) = 0;
    virtual const std::shared_ptr<Fillable> endrecord() = 0;

    void setthat(const std::shared_ptr<Fillable>& that);

  protected:
    std::shared_ptr<Fillable> that_;
  };
}

#endif // AWKWARD_FILLABLE_H_

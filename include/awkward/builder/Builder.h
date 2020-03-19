// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include <string>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class EXPORT_SYMBOL Builder {
  public:
    virtual ~Builder();

    virtual const std::string classname() const = 0;
    virtual int64_t length() const = 0;
    virtual void clear() = 0;
    virtual ContentPtr snapshot() const = 0;

    virtual bool active() const = 0;
    virtual const std::shared_ptr<Builder> null() = 0;
    virtual const std::shared_ptr<Builder> boolean(bool x) = 0;
    virtual const std::shared_ptr<Builder> integer(int64_t x) = 0;
    virtual const std::shared_ptr<Builder> real(double x) = 0;
    virtual const std::shared_ptr<Builder> string(const char* x, int64_t length, const char* encoding) = 0;
    virtual const std::shared_ptr<Builder> beginlist() = 0;
    virtual const std::shared_ptr<Builder> endlist() = 0;
    virtual const std::shared_ptr<Builder> begintuple(int64_t numfields) = 0;
    virtual const std::shared_ptr<Builder> index(int64_t index) = 0;
    virtual const std::shared_ptr<Builder> endtuple() = 0;
    virtual const std::shared_ptr<Builder> beginrecord(const char* name, bool check) = 0;
    virtual const std::shared_ptr<Builder> field(const char* key, bool check) = 0;
    virtual const std::shared_ptr<Builder> endrecord() = 0;
    virtual const std::shared_ptr<Builder> append(ContentPtr& array, int64_t at) = 0;

    void setthat(const std::shared_ptr<Builder>& that);

  protected:
    std::shared_ptr<Builder> that_;
  };
}

#endif // AWKWARD_FILLABLE_H_

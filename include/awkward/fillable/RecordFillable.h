// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDFILLABLE_H_
#define AWKWARD_RECORDFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  class RecordFillable: public Fillable {
  public:
    RecordFillable(const FillableOptions& options, const std::vector<std::shared_ptr<Fillable>>& contents, const std::vector<std::string>& keys, const std::vector<const char*>& pointers, int64_t disambiguator, int64_t length, bool begun, int64_t nextindex, int64_t nexttotry)
        : options_(options)
        , contents_(contents)
        , keys_(keys)
        , pointers_(pointers)
        , disambiguator_(disambiguator)
        , length_(length)
        , begun_(begun)
        , nextindex_(nextindex)
        , nexttotry_(nexttotry) { }

    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options);

    virtual const std::string classname() const { return "RecordFillable"; };
    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual const std::shared_ptr<Fillable> null();
    virtual const std::shared_ptr<Fillable> boolean(bool x);
    virtual const std::shared_ptr<Fillable> integer(int64_t x);
    virtual const std::shared_ptr<Fillable> real(double x);
    virtual const std::shared_ptr<Fillable> beginlist();
    virtual const std::shared_ptr<Fillable> endlist();
    virtual const std::shared_ptr<Fillable> begintuple(int64_t numfields);
    virtual const std::shared_ptr<Fillable> index(int64_t index);
    virtual const std::shared_ptr<Fillable> endtuple();
    virtual const std::shared_ptr<Fillable> beginrecord(int64_t disambiguator);
    virtual const std::shared_ptr<Fillable> field_fast(const char* key);
    virtual const std::shared_ptr<Fillable> field_check(const char* key);
    virtual const std::shared_ptr<Fillable> endrecord();

    int64_t disambiguator() const { return disambiguator_; }

  private:
    const FillableOptions options_;
    std::vector<std::shared_ptr<Fillable>> contents_;
    std::vector<std::string> keys_;
    std::vector<const char*> pointers_;
    int64_t disambiguator_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;
    int64_t nexttotry_;

    void maybeupdate(int64_t i, const std::shared_ptr<Fillable>& tmp);
  };
}

#endif // AWKWARD_RECORDFILLABLE_H_

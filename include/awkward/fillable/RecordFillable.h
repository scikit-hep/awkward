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
    RecordFillable(const FillableOptions& options, const std::vector<std::shared_ptr<Fillable>>& contents, const std::vector<std::string>& keys, int64_t disambiguator, int64_t length, bool begun, size_t nextindex): options_(options), contents_(contents), keys_(keys), disambiguator_(disambiguator), length_(length), begun_(begun), nextindex_(nextindex) { }

    static RecordFillable* fromempty(const FillableOptions& options) {
      return new RecordFillable(options, std::vector<std::shared_ptr<Fillable>>(), std::vector<std::string>(), 0, -1, false, -1);
    }


    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual Fillable* null();
    virtual Fillable* boolean(bool x);
    virtual Fillable* integer(int64_t x);
    virtual Fillable* real(double x);
    virtual Fillable* beginlist();
    virtual Fillable* endlist();
    virtual Fillable* begintuple(int64_t numfields);
    virtual Fillable* index(int64_t index);
    virtual Fillable* endtuple();
    virtual Fillable* beginrecord(int64_t disambiguator);
    virtual Fillable* field_fast(const char* key);
    virtual Fillable* field_check(const char* key);
    virtual Fillable* endrecord();

    int64_t disambiguator() const { return disambiguator_; }

  private:
    const FillableOptions options_;
    std::vector<std::shared_ptr<Fillable>> contents_;
    std::vector<std::string> keys_;
    int64_t disambiguator_;
    int64_t length_;
    bool begun_;
    size_t nextindex_;

    void maybeupdate(int64_t i, Fillable* tmp);
  };
}

#endif // AWKWARD_RECORDFILLABLE_H_

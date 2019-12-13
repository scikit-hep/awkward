// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_STRINGFILLABLE_H_
#define AWKWARD_STRINGFILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  class StringFillable: public Fillable {
  public:
    StringFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, GrowableBuffer<uint8_t>& content, const char* encoding): options_(options), offsets_(offsets), content_(content), encoding_(encoding) { }

    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options, const char* encoding);

    virtual const std::string classname() const { return "StringFillable"; };
    virtual int64_t length() const;
    virtual void clear();
    virtual const std::shared_ptr<Type> type() const;
    virtual const std::shared_ptr<Content> snapshot() const;

    virtual bool active() const;
    virtual const std::shared_ptr<Fillable> null();
    virtual const std::shared_ptr<Fillable> boolean(bool x);
    virtual const std::shared_ptr<Fillable> integer(int64_t x);
    virtual const std::shared_ptr<Fillable> real(double x);
    virtual const std::shared_ptr<Fillable> string(const char* x, int64_t length, const char* encoding);
    virtual const std::shared_ptr<Fillable> beginlist();
    virtual const std::shared_ptr<Fillable> endlist();
    virtual const std::shared_ptr<Fillable> begintuple(int64_t numfields);
    virtual const std::shared_ptr<Fillable> index(int64_t index);
    virtual const std::shared_ptr<Fillable> endtuple();
    virtual const std::shared_ptr<Fillable> beginrecord(const char* name, bool check);
    virtual const std::shared_ptr<Fillable> field(const char* key, bool check);
    virtual const std::shared_ptr<Fillable> endrecord();

    const char* encoding() const { return encoding_; }

  private:
    const FillableOptions options_;
    GrowableBuffer<int64_t> offsets_;
    GrowableBuffer<uint8_t> content_;
    const char* encoding_;
  };
}

#endif // AWKWARD_STRINGFILLABLE_H_

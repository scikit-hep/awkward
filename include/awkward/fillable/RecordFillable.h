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
    static const std::shared_ptr<Fillable> fromempty(const FillableOptions& options);

    RecordFillable(const FillableOptions& options, const std::vector<std::shared_ptr<Fillable>>& contents, const std::vector<std::string>& keys, const std::vector<const char*>& pointers, const std::string& name, const char* nameptr, int64_t length, bool begun, int64_t nextindex, int64_t nexttotry);

    const std::string name() const;
    const char* nameptr() const;

    const std::string classname() const override;
    int64_t length() const override;
    void clear() override;
    const std::shared_ptr<Content> snapshot() const override;

    bool active() const override;
    const std::shared_ptr<Fillable> null() override;
    const std::shared_ptr<Fillable> boolean(bool x) override;
    const std::shared_ptr<Fillable> integer(int64_t x) override;
    const std::shared_ptr<Fillable> real(double x) override;
    const std::shared_ptr<Fillable> string(const char* x, int64_t length, const char* encoding) override;
    const std::shared_ptr<Fillable> beginlist() override;
    const std::shared_ptr<Fillable> endlist() override;
    const std::shared_ptr<Fillable> begintuple(int64_t numfields) override;
    const std::shared_ptr<Fillable> index(int64_t index) override;
    const std::shared_ptr<Fillable> endtuple() override;
    const std::shared_ptr<Fillable> beginrecord(const char* name, bool check) override;
    const std::shared_ptr<Fillable> field(const char* key, bool check) override;
    const std::shared_ptr<Fillable> endrecord() override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;

  private:
    const std::shared_ptr<Fillable> field_fast(const char* key);
    const std::shared_ptr<Fillable> field_check(const char* key);

    const FillableOptions options_;
    std::vector<std::shared_ptr<Fillable>> contents_;
    std::vector<std::string> keys_;
    std::vector<const char*> pointers_;
    std::string name_;
    const char* nameptr_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;
    int64_t nexttotry_;

    void maybeupdate(int64_t i, const std::shared_ptr<Fillable>& tmp);
  };
}

#endif // AWKWARD_RECORDFILLABLE_H_

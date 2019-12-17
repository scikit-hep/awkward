// BSD 3-Clause License; see
// https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_OPTIONFILLABLE_H_
#define AWKWARD_OPTIONFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/Fillable.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"

namespace awkward {
class OptionFillable : public Fillable {
public:
  OptionFillable(const FillableOptions &options,
                 const GrowableBuffer<int64_t> &offsets,
                 std::shared_ptr<Fillable> content)
      : options_(options), offsets_(offsets), content_(content) {}

  static const std::shared_ptr<Fillable>
  fromnulls(const FillableOptions &options, int64_t nullcount,
            std::shared_ptr<Fillable> content);
  static const std::shared_ptr<Fillable>
  fromvalids(const FillableOptions &options, std::shared_ptr<Fillable> content);

  virtual const std::string classname() const { return "OptionFillable"; };
  virtual int64_t length() const;
  virtual void clear();
  virtual const std::shared_ptr<Type> type() const;
  virtual const std::shared_ptr<Content> snapshot() const;

  virtual bool active() const;
  virtual const std::shared_ptr<Fillable> null();
  virtual const std::shared_ptr<Fillable> boolean(bool x);
  virtual const std::shared_ptr<Fillable> integer(int64_t x);
  virtual const std::shared_ptr<Fillable> real(double x);
  virtual const std::shared_ptr<Fillable> string(const char *x, int64_t length,
                                                 const char *encoding);
  virtual const std::shared_ptr<Fillable> beginlist();
  virtual const std::shared_ptr<Fillable> endlist();
  virtual const std::shared_ptr<Fillable> begintuple(int64_t numfields);
  virtual const std::shared_ptr<Fillable> index(int64_t index);
  virtual const std::shared_ptr<Fillable> endtuple();
  virtual const std::shared_ptr<Fillable> beginrecord(const char *name,
                                                      bool check);
  virtual const std::shared_ptr<Fillable> field(const char *key, bool check);
  virtual const std::shared_ptr<Fillable> endrecord();

private:
  const FillableOptions options_;
  GrowableBuffer<int64_t> offsets_;
  std::shared_ptr<Fillable> content_;

  void maybeupdate(const std::shared_ptr<Fillable> &tmp);
};
} // namespace awkward

#endif // AWKWARD_OPTIONFILLABLE_H_

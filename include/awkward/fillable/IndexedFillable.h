// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEXEDFILLABLE_H_
#define AWKWARD_INDEXEDFILLABLE_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/fillable/FillableOptions.h"
#include "awkward/fillable/GrowableBuffer.h"
#include "awkward/fillable/Fillable.h"

namespace awkward {
  template <typename T>
  class EXPORT_SYMBOL IndexedFillable: public Fillable {
  public:
    IndexedFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<T>& array, bool hasnull);

    const Content* arrayptr() const;

    int64_t length() const override;
    void clear() override;

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

  protected:
    const FillableOptions options_;
    GrowableBuffer<int64_t> index_;
    const std::shared_ptr<T> array_;
    bool hasnull_;
  };

  class EXPORT_SYMBOL IndexedGenericFillable: public IndexedFillable<Content> {
  public:
    static const std::shared_ptr<Fillable> fromnulls(const FillableOptions& options, int64_t nullcount, const std::shared_ptr<Content>& array);

    IndexedGenericFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<Content>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class EXPORT_SYMBOL IndexedI32Fillable: public IndexedFillable<IndexedArray32> {
  public:
    IndexedI32Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray32>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class EXPORT_SYMBOL IndexedIU32Fillable: public IndexedFillable<IndexedArrayU32> {
  public:
    IndexedIU32Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArrayU32>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class EXPORT_SYMBOL IndexedI64Fillable: public IndexedFillable<IndexedArray64> {
  public:
    IndexedI64Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray64>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class EXPORT_SYMBOL IndexedIO32Fillable: public IndexedFillable<IndexedOptionArray32> {
  public:
    IndexedIO32Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray32>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class EXPORT_SYMBOL IndexedIO64Fillable: public IndexedFillable<IndexedOptionArray64> {
  public:
    IndexedIO64Fillable(const FillableOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray64>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Fillable> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

}

#endif // AWKWARD_INDEXEDFILLABLE_H_

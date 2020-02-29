// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEXEDBUILDER_H_
#define AWKWARD_INDEXEDBUILDER_H_

#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  template <typename T>
  class IndexedBuilder: public Builder {
  public:
    IndexedBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<T>& array, bool hasnull);

    const Content* arrayptr() const;

    int64_t length() const override;
    void clear() override;

    bool active() const override;
    const std::shared_ptr<Builder> null() override;
    const std::shared_ptr<Builder> boolean(bool x) override;
    const std::shared_ptr<Builder> integer(int64_t x) override;
    const std::shared_ptr<Builder> real(double x) override;
    const std::shared_ptr<Builder> string(const char* x, int64_t length, const char* encoding) override;
    const std::shared_ptr<Builder> beginlist() override;
    const std::shared_ptr<Builder> endlist() override;
    const std::shared_ptr<Builder> begintuple(int64_t numfields) override;
    const std::shared_ptr<Builder> index(int64_t index) override;
    const std::shared_ptr<Builder> endtuple() override;
    const std::shared_ptr<Builder> beginrecord(const char* name, bool check) override;
    const std::shared_ptr<Builder> field(const char* key, bool check) override;
    const std::shared_ptr<Builder> endrecord() override;

  protected:
    const ArrayBuilderOptions options_;
    GrowableBuffer<int64_t> index_;
    const std::shared_ptr<T> array_;
    bool hasnull_;
  };

  class IndexedGenericBuilder: public IndexedBuilder<Content> {
  public:
    static const std::shared_ptr<Builder> fromnulls(const ArrayBuilderOptions& options, int64_t nullcount, const std::shared_ptr<Content>& array);

    IndexedGenericBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<Content>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Builder> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class IndexedI32Builder: public IndexedBuilder<IndexedArray32> {
  public:
    IndexedI32Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray32>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Builder> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class IndexedIU32Builder: public IndexedBuilder<IndexedArrayU32> {
  public:
    IndexedIU32Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArrayU32>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Builder> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class IndexedI64Builder: public IndexedBuilder<IndexedArray64> {
  public:
    IndexedI64Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedArray64>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Builder> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class IndexedIO32Builder: public IndexedBuilder<IndexedOptionArray32> {
  public:
    IndexedIO32Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray32>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Builder> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

  class IndexedIO64Builder: public IndexedBuilder<IndexedOptionArray64> {
  public:
    IndexedIO64Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& index, const std::shared_ptr<IndexedOptionArray64>& array, bool hasnull);

    const std::string classname() const override;
    const std::shared_ptr<Content> snapshot() const override;
    const std::shared_ptr<Builder> append(const std::shared_ptr<Content>& array, int64_t at) override;
  };

}

#endif // AWKWARD_INDEXEDBUILDER_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NUMPYARRAY_H_
#define AWKWARD_NUMPYARRAY_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  class EXPORT_SYMBOL NumpyArray: public Content {
  public:
    static const std::shared_ptr<Type> unwrap_regulartype(const std::shared_ptr<Type>& type, const std::vector<ssize_t>& shape);
    static const std::unordered_map<std::type_index, std::string> format_map;

    NumpyArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::shared_ptr<void>& ptr, const std::vector<ssize_t>& shape, const std::vector<ssize_t>& strides, ssize_t byteoffset, ssize_t itemsize, const std::string format);
    NumpyArray(const Index8 index);
    NumpyArray(const IndexU8 index);
    NumpyArray(const Index32 index);
    NumpyArray(const IndexU32 index);
    NumpyArray(const Index64 index);
    NumpyArray(const Index8 index, const std::string& format);
    NumpyArray(const IndexU8 index, const std::string& format);
    NumpyArray(const Index32 index, const std::string& format);
    NumpyArray(const IndexU32 index, const std::string& format);
    NumpyArray(const Index64 index, const std::string& format);

    const std::shared_ptr<void> ptr() const;
    const std::vector<ssize_t> shape() const;
    const std::vector<ssize_t> strides() const;
    ssize_t byteoffset() const;
    ssize_t itemsize() const;
    const std::string format() const;

    ssize_t ndim() const;
    bool isempty() const;
    void* byteptr() const;
    void* byteptr(ssize_t at) const;
    ssize_t bytelength() const;
    uint8_t getbyte(ssize_t at) const;
    int8_t getint8(ssize_t at) const;
    uint8_t getuint8(ssize_t at) const;
    int16_t getint16(ssize_t at) const;
    uint16_t getuint16(ssize_t at) const;
    int32_t getint32(ssize_t at) const;
    uint32_t getuint32(ssize_t at) const;
    int64_t getint64(ssize_t at) const;
    uint64_t getuint64(ssize_t at) const;
    float_t getfloat(ssize_t at) const;
    double_t getdouble(ssize_t at) const;
    const std::shared_ptr<Content> toRegularArray() const;

    bool isscalar() const override;
    const std::string classname() const override;
    void setidentities() override;
    void setidentities(const std::shared_ptr<Identities>& identities) override;
    const std::shared_ptr<Type> type(const std::map<std::string, std::string>& typestrs) const override;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    void tojson_part(ToJson& builder) const override;
    void nbytes_part(std::map<size_t, int64_t>& largest) const override;
    int64_t length() const override;
    const std::shared_ptr<Content> shallow_copy() const override;
    const std::shared_ptr<Content> deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const override;
    void check_for_iteration() const override;
    const std::shared_ptr<Content> getitem_nothing() const override;
    const std::shared_ptr<Content> getitem_at(int64_t at) const override;
    const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const override;
    const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const override;
    const std::shared_ptr<Content> getitem_field(const std::string& key) const override;
    const std::shared_ptr<Content> getitem_fields(const std::vector<std::string>& keys) const override;
    const std::shared_ptr<Content> getitem(const Slice& where) const override;
    const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> carry(const Index64& carry) const override;
    const std::string purelist_parameter(const std::string& key) const override;
    bool purelist_isregular() const override;
    int64_t purelist_depth() const override;
    const std::pair<int64_t, int64_t> minmax_depth() const override;
    const std::pair<bool, int64_t> branch_depth() const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    // operations
    const std::string validityerror(const std::string& path) const override;
    const std::shared_ptr<Content> shallow_simplify() const override;
    const std::shared_ptr<Content> num(int64_t axis, int64_t depth) const override;
    const std::pair<Index64, std::shared_ptr<Content>> offsets_and_flattened(int64_t axis, int64_t depth) const override;
    bool mergeable(const std::shared_ptr<Content>& other, bool mergebool) const override;
    const std::shared_ptr<Content> merge(const std::shared_ptr<Content>& other) const override;
    const std::shared_ptr<SliceItem> asslice() const override;
    const std::shared_ptr<Content> rpad(int64_t length, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& parents, int64_t outlength, bool mask, bool keepdims, bool semigroup) const override;
    const std::shared_ptr<Content> localindex(int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const override;

    bool iscontiguous() const;
    const NumpyArray contiguous() const;

    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const override;

  protected:
    const NumpyArray contiguous_next(const Index64& bytepos) const;
    const NumpyArray getitem_bystrides(const std::shared_ptr<SliceItem>& head, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceAt& at, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceRange& range, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceEllipsis& ellipsis, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_bystrides(const SliceNewAxis& newaxis, const Slice& tail, int64_t length) const;
    const NumpyArray getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceAt& at, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceRange& range, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceEllipsis& ellipsis, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceNewAxis& newaxis, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const NumpyArray getitem_next(const SliceArray64& array, const Slice& tail, const Index64& carry, const Index64& advanced, int64_t length, int64_t stride, bool first) const;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const override;

  void tojson_boolean(ToJson& builder) const;
  template <typename T>
  void tojson_integer(ToJson& builder) const;
  template <typename T>
  void tojson_real(ToJson& builder) const;
  void tojson_string(ToJson& builder) const;

  private:
    std::shared_ptr<void> ptr_;
    std::vector<ssize_t> shape_;
    std::vector<ssize_t> strides_;
    ssize_t byteoffset_;
    const ssize_t itemsize_;
    const std::string format_;
  };
}

#endif // AWKWARD_NUMPYARRAY_H_

// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NONE_H_
#define AWKWARD_NONE_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class None
  ///
  /// @brief Represents a scalar missing value, which is `None` in Python.
  ///
  /// Nearly all of the methods raise runtime errors. See
  /// {@link Content#isscalar Content::isscalar}.
  class EXPORT_SYMBOL None: public Content {
  public:
    /// @brief Creates a None instance.
    None();

    /// @copydoc Content::isscalar()
    ///
    /// Always returns `true`.
    bool
      isscalar() const override;

    /// @brief User-friendly name of this class: `"None"`.
    const std::string
      classname() const override;

    /// @exception std::runtime_error is always thrown
    void
      setidentities() override;

    /// @exception std::runtime_error is always thrown
    void
      setidentities(const IdentitiesPtr& identities) override;

    /// @exception std::runtime_error is always thrown
    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    void
      tojson_part(ToJson& builder) const override;

    /// @exception std::runtime_error is always thrown
    void
      nbytes_part(std::map<size_t, int64_t>& largest) const override;

    /// Always returns `-1`.
    int64_t
      length() const override;

    const ContentPtr
      shallow_copy() const override;

    const ContentPtr
      deep_copy(bool copyarrays,
                bool copyindexes,
                bool copyidentities) const override;

    void
      check_for_iteration() const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_nothing() const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_at(int64_t at) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_at_nowrap(int64_t at) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_field(const std::string& key) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      carry(const Index64& carry) const override;

    /// @exception std::runtime_error is always thrown
    const std::string
      purelist_parameter(const std::string& key) const override;

    /// @exception std::runtime_error is always thrown
    bool
      purelist_isregular() const override;

    /// @exception std::runtime_error is always thrown
    int64_t
      purelist_depth() const override;

    /// @exception std::runtime_error is always thrown
    const std::pair<int64_t, int64_t>
      minmax_depth() const override;

    /// @exception std::runtime_error is always thrown
    const std::pair<bool, int64_t>
      branch_depth() const override;

    /// @exception std::runtime_error is always thrown
    int64_t
      numfields() const override;

    /// @exception std::runtime_error is always thrown
    int64_t
      fieldindex(const std::string& key) const override;

    /// @exception std::runtime_error is always thrown
    const std::string
      key(int64_t fieldindex) const override;

    /// @exception std::runtime_error is always thrown
    bool
      haskey(const std::string& key) const override;

    /// @exception std::runtime_error is always thrown
    const std::vector<std::string>
      keys() const override;

    // operations
<<<<<<< HEAD
    const std::string validityerror(const std::string& path) const override;
    const std::shared_ptr<Content> shallow_simplify() const override;
    const std::shared_ptr<Content> num(int64_t axis, int64_t depth) const override;
    const std::pair<Index64, std::shared_ptr<Content>> offsets_and_flattened(int64_t axis, int64_t depth) const override;
    bool mergeable(const std::shared_ptr<Content>& other, bool mergebool) const override;
    const std::shared_ptr<Content> merge(const std::shared_ptr<Content>& other) const override;
    const std::shared_ptr<SliceItem> asslice() const override;
    const std::shared_ptr<Content> rpad(int64_t length, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& starts, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const override;
    const std::shared_ptr<Content> localindex(int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const override;
    const std::shared_ptr<Content> sort_next(int64_t negaxis, const Index64& starts, const Index64& parents, int64_t outlength, bool ascending, bool stable) const override;

    const std::shared_ptr<Content> getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const override;
    const std::shared_ptr<Content> getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const override;
=======

    /// @exception std::runtime_error is always thrown
    const std::string
      validityerror(const std::string& path) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      shallow_simplify() const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      num(int64_t axis, int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    bool
      mergeable(const ContentPtr& other, bool mergebool) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      merge(const ContentPtr& other) const override;

    /// @exception std::runtime_error is always thrown
    const SliceItemPtr
      asslice() const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      fillna(const ContentPtr& value) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      rpad(int64_t target, int64_t axis, int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      rpad_and_clip(int64_t target,
                    int64_t axis,
                    int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      reduce_next(const Reducer& reducer,
                  int64_t negaxis,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength,
                  bool mask,
                  bool keepdims) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      sort_next(int64_t negaxis,
                const Index64& starts,
                const Index64& parents,
                int64_t outlength,
                bool ascending,
                bool stable) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      localindex(int64_t axis, int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      choose(int64_t n,
             bool diagonal,
             const util::RecordLookupPtr& recordlookup,
             const util::Parameters& parameters,
             int64_t axis,
             int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next(const SliceField& field,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next(const SliceFields& fields,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next(const SliceJagged64& jagged,
                   const Slice& tail,
                   const Index64& advanced) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceArray64& slicecontent,
                          const Slice& tail) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceMissing64& slicecontent,
                          const Slice& tail) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceJagged64& slicecontent,
                          const Slice& tail) const override;
>>>>>>> a4211f2338b89fafbe926d21e9e4b798d504ef84
  };

  /// A constant value with type None.
  extern const ContentPtr none;
}

#endif // AWKWARD_NONE_H_

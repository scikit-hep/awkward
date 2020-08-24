// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARARRAY_H_
#define AWKWARD_REGULARARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class RegularForm
  ///
  /// @brief Form describing RegularArray.
  class LIBAWKWARD_EXPORT_SYMBOL RegularForm: public Form {
  public:
    /// @brief Creates a RegularForm. See RegularArray for documentation.
    RegularForm(bool has_identities,
                const util::Parameters& parameters,
                const FormKey& form_key,
                const FormPtr& content,
                int64_t size);

    const FormPtr
      content() const;

    int64_t
      size() const;

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    void
      tojson_part(ToJson& builder, bool verbose) const override;

    const FormPtr
      shallow_copy() const override;

    const std::string
      purelist_parameter(const std::string& key) const override;

    bool
      purelist_isregular() const override;

    int64_t
      purelist_depth() const override;

    const std::pair<int64_t, int64_t>
      minmax_depth() const override;

    const std::pair<bool, int64_t>
      branch_depth() const override;

    int64_t
      numfields() const override;

    int64_t
      fieldindex(const std::string& key) const override;

    const std::string
      key(int64_t fieldindex) const override;

    bool
      haskey(const std::string& key) const override;

    const std::vector<std::string>
      keys() const override;

    bool
      equal(const FormPtr& other,
            bool check_identities,
            bool check_parameters,
            bool check_form_key,
            bool compatibility_check) const override;

    const FormPtr
      getitem_field(const std::string& key) const override;

  private:
    const FormPtr content_;
    int64_t size_;
  };

  /// @class RegularArray
  ///
  /// @brief Represents an array of nested lists that all have the same
  /// length using a scalar #size, rather than an index.
  ///
  /// The #content must be contiguous, in-order, and non-overlapping, and
  /// it must also start at zero.
  ///
  /// See #RegularArray for the meaning of each parameter.
  class LIBAWKWARD_EXPORT_SYMBOL RegularArray: public Content {
  public:
    /// @brief Creates a RegularArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param content Data contained within all nested lists as a contiguous
    /// array.
    /// Values in `content[i]` where `i >= length * size` are "unreachable,"
    /// and don't exist in the high level view.
    /// @param size Length of the equally sized nested lists.
    RegularArray(const IdentitiesPtr& identities,
                 const util::Parameters& parameters,
                 const ContentPtr& content,
                 int64_t size);

    /// @brief Data contained within all nested lists as a contiguous array.
    ///
    /// Values in `content[i]` where `i >= length * size` are "unreachable,"
    /// and don't exist in the high level view.
    const ContentPtr
      content() const;

    /// @brief Length of the equally sized nested lists.
    int64_t
      size() const;

    /// @brief Returns 64-bit offsets, possibly starting with `offsets[0] = 0`,
    /// that would represent the spacing in this RegularArray.
    ///
    /// @param start_at_zero If `true`, the first offset will be `0`, meaning
    /// there are no "unreachable" elements in the `content` that corresponds
    /// to these offsets.
    Index64
      compact_offsets64(bool start_at_zero) const;

    /// @brief Verify that a given set of `offsets` are regular
    /// and return a {@link ListOffsetArrayOf ListOffsetArray} of this array
    /// using those `offsets`.
    ///
    /// As indicated by the name, this is a basic element of broadcasting.
    const ContentPtr
      broadcast_tooffsets64(const Index64& offsets) const;

    /// @brief Effectively the same as #shallow_copy, but with the same name
    /// as the equivalent
    /// {@link ListArrayOf#toRegularArray ListArray::toRegularArray} and
    /// {@link ListOffsetArrayOf#toRegularArray ListOffsetArray::toRegularArray}.
    const ContentPtr
      toRegularArray() const;

    /// @brief Converts this array into a
    /// {@link ListOffsetArrayOf ListOffsetArray} by generating `offsets` that
    /// are equivalent to its regular #size.
    ///
    /// @param start_at_zero If `true`, the first offset will be `0`, meaning
    /// there are no "unreachable" elements in the `content` that corresponds
    /// to these offsets. For a RegularArray, this would always be true.
    const ContentPtr
      toListOffsetArray64(bool start_at_zero) const;

    /// @brief User-friendly name of this class: `"RegularArray"`.
    const std::string
      classname() const override;

    void
      setidentities() override;

    void
      setidentities(const IdentitiesPtr& identities) override;

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    const FormPtr
      form(bool materialize) const override;

    bool
      has_virtual_form() const override;

    bool
      has_virtual_length() const override;

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    void
      tojson_part(ToJson& builder, bool include_beginendlist) const override;

    void
      nbytes_part(std::map<size_t, int64_t>& largest) const override;

    /// @copydoc Content::length()
    ///
    /// Equal to `floor(len(content) / size)`.
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

    const ContentPtr
      getitem_nothing() const override;

    const ContentPtr
      getitem_at(int64_t at) const override;

    const ContentPtr
      getitem_at_nowrap(int64_t at) const override;

    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const override;

    const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const override;

    const ContentPtr
      getitem_field(const std::string& key) const override;

    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const override;

    const ContentPtr
      carry(const Index64& carry, bool allow_lazy) const override;

    int64_t
      numfields() const override;

    int64_t
      fieldindex(const std::string& key) const override;

    const std::string
      key(int64_t fieldindex) const override;

    bool
      haskey(const std::string& key) const override;

    const std::vector<std::string>
      keys() const override;

    // operations
    const std::string
      validityerror(const std::string& path) const override;

    /// @copydoc Content::shallow_simplify()
    ///
    /// For RegularArray, this method returns #shallow_copy (pass-through).
    const ContentPtr
      shallow_simplify() const override;

    const ContentPtr
      num(int64_t axis, int64_t depth) const override;

    const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const override;

    bool
      mergeable(const ContentPtr& other, bool mergebool) const override;

    const ContentPtr
      merge(const ContentPtr& other) const override;

    const SliceItemPtr
      asslice() const override;

    const ContentPtr
      fillna(const ContentPtr& value) const override;

    const ContentPtr
      rpad(int64_t target, int64_t axis, int64_t depth) const override;

    const ContentPtr
      rpad_and_clip(int64_t target,
                    int64_t axis,
                    int64_t depth) const override;

    const ContentPtr
      reduce_next(const Reducer& reducer,
                  int64_t negaxis,
                  const Index64& starts,
                  const Index64& shifts,
                  const Index64& parents,
                  int64_t outlength,
                  bool mask,
                  bool keepdims) const override;

    const ContentPtr
      sort_next(int64_t negaxis,
                const Index64& starts,
                const Index64& parents,
                int64_t outlength,
                bool ascending,
                bool stable,
                bool keepdims) const override;

    const ContentPtr
      argsort_next(int64_t negaxis,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength,
                   bool ascending,
                   bool stable,
                   bool keepdims) const override;

    const ContentPtr
      localindex(int64_t axis, int64_t depth) const override;

    const ContentPtr
      combinations(int64_t n,
                   bool replacement,
                   const util::RecordLookupPtr& recordlookup,
                   const util::Parameters& parameters,
                   int64_t axis,
                   int64_t depth) const override;

    const ContentPtr
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& advanced) const override;

    const ContentPtr
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& advanced) const override;

    const ContentPtr
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& advanced) const override;

    const ContentPtr
      getitem_next(const SliceJagged64& jagged,
                   const Slice& tail,
                   const Index64& advanced) const override;

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceArray64& slicecontent,
                          const Slice& tail) const override;

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceMissing64& slicecontent,
                          const Slice& tail) const override;

    const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceJagged64& slicecontent,
                          const Slice& tail) const override;

    const ContentPtr
      copy_to(kernel::lib ptr_lib) const override;

    const ContentPtr
      numbers_to_type(const std::string& name) const override;

  private:
    const ContentPtr content_;
    int64_t size_;
  };
}

#endif // AWKWARD_REGULARARRAY_H_

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTARRAY_H_
#define AWKWARD_LISTARRAY_H_

#include <memory>

#include "awkward/common.h"
#include "awkward/Index.h"
#include "awkward/Identities.h"
#include "awkward/Content.h"
#include "awkward/kernel-dispatch.h"

namespace awkward {
  /// @class ListForm
  ///
  /// @brief Form describing ListArray.
  class LIBAWKWARD_EXPORT_SYMBOL ListForm: public Form {
  public:
    /// @brief Creates a ListForm. See {@link ListArrayOf LinkArray} for
    /// documentation.
    ListForm(bool has_identities,
             const util::Parameters& parameters,
             const FormKey& form_key,
             Index::Form starts,
             Index::Form stops,
             const FormPtr& content);

    Index::Form
      starts() const;

    Index::Form
      stops() const;

    const FormPtr
      content() const;

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
    Index::Form starts_;
    Index::Form stops_;
    const FormPtr content_;
  };

  /// @class ListArrayOf
  ///
  /// @brief Represents an array of nested lists that can have different
  /// lengths using two indexes named #starts and #stops.
  ///
  /// The use of two indexes, #starts and #stops, allows the #content to be
  /// non-contiguous, out-of-order, and possibly overlapping.
  ///
  /// See #ListArrayOf for the meaning of each parameter.
  template <typename T>
  class
#ifdef AWKWARD_LISTARRAY_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  ListArrayOf: public Content {
  public:
    /// @brief Creates a ListArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param starts Positions where each nested list starts in the #content.
    /// @param stops Positions where each nested list stops in the #content.
    /// The `starts` and `stops` may be in any order, they may repeat elements,
    /// may represent partially or completely overlapping ranges of the
    /// #content, and they may leave "unreachable" gaps between lists.
    /// If `starts[i] == stops[i]`, there is no constraint on the value of
    /// `starts[i]`. Otherwise, `0 <= starts[i] < len(content)` and
    /// `0 <= stops[i] <= len(content)`.
    /// @param content Data referenced by the #starts and #stops to build
    /// nested lists.
    /// The `content` does not necessarily represent a flattened version of
    /// this array because a single element may belong to multiple lists or
    /// no list at all.
    ListArrayOf<T>(const IdentitiesPtr& identities,
                   const util::Parameters& parameters,
                   const IndexOf<T>& starts,
                   const IndexOf<T>& stops,
                   const ContentPtr& content);

    /// @brief Positions where each nested list starts in the #content.
    ///
    /// The `starts` may be in any order, they may repeat elements, may
    /// represent partially or completely overlapping ranges of the #content,
    /// and they may leave "unreachable" gaps between lists.
    ///
    /// If `starts[i] == stops[i]`, there is no constraint on the value of
    /// `starts[i]`. Otherwise, `0 <= starts[i] < len(content)`.
    const IndexOf<T>
      starts() const;

    /// @brief Positions where each nested list stops in the #content.
    ///
    /// The `stops` may be in any order, they may repeat elements, may
    /// represent partially or completely overlapping ranges of the #content,
    /// and they may leave "unreachable" gaps between lists.
    ///
    /// If `starts[i] == stops[i]`, there is no constraint on the value of
    /// `stops[i]`. Otherwise, `0 <= stops[i] <= len(content)`.
    const IndexOf<T>
      stops() const;

    /// @brief Data referenced by the #starts and #stops to build nested lists.
    ///
    /// The `content` does not necessarily represent a flattened version of
    /// this array because a single element may belong to multiple lists or
    /// no list at all.
    const ContentPtr
      content() const;

    /// @brief Returns 64-bit offsets, possibly starting with `offsets[0] = 0`,
    /// that would represent this array's #starts and #stops if the #content
    /// were replaced by a contiguous copy.
    ///
    /// @param start_at_zero If `true`, the first offset will be `0`, meaning
    /// there are no "unreachable" elements in the `content` that corresponds
    /// to these offsets.
    Index64
      compact_offsets64(bool start_at_zero) const;

    /// @brief Moves #content elements if necessary to match a given set of
    /// `offsets` and return a {@link ListOffsetArrayOf ListOffsetArray} that
    /// matches.
    ///
    /// As indicated by the name, this is a basic element of broadcasting.
    ///
    /// Since the output is a {@link ListOffsetArrayOf ListOffsetArray}, this
    /// operation produces contiguous output, replacing multiply-referenced
    /// items with copied items and removing unreachable gaps between items.
    const ContentPtr
      broadcast_tooffsets64(const Index64& offsets) const;

    /// @brief Converts this array to a RegularArray if all nested lists have
    /// the same size (error otherwise).
    ///
    /// Since the output is a RegularArray, this
    /// operation produces contiguous output, replacing multiply-referenced
    /// items with copied items and removing unreachable gaps before and
    /// between items.
    const ContentPtr
      toRegularArray() const;

    /// @brief Returns this array as a
    /// {@link ListOffsetArrayOf ListOffsetArray} with
    /// 64-bit {@link ListOffsetArrayOf#offsets offsets} and possibly starting
    ///  with `offsets[0] = 0`; a #shallow_copy if possible.
    ///
    /// @param start_at_zero If `true`, the first offset will be `0`, meaning
    /// there are no "unreachable" elements in the `content` that corresponds
    /// to these offsets.
    ///
    /// Since the output is a {@link ListOffsetArrayOf ListOffsetArray}, this
    /// operation produces contiguous output, replacing multiply-referenced
    /// items with copied items and removing unreachable gaps between items.
    const ContentPtr
      toListOffsetArray64(bool start_at_zero) const;

    /// @brief User-friendly name of this class: `"ListArray32"`,
    /// `"ListArrayU32"`, or `"ListArray64"`.
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
    /// Equal to `len(starts)`.
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
    /// For {@link ListArrayOf ListArray}, this method returns
    /// #shallow_copy (pass-through).
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
    /// @brief See #starts.
    const IndexOf<T> starts_;
    /// @brief See #stops.
    const IndexOf<T> stops_;
    /// @brief See #content.
    const ContentPtr content_;
  };

#ifndef AWKWARD_LISTARRAY_NO_EXTERN_TEMPLATE
  extern template class ListArrayOf<int32_t>;
  extern template class ListArrayOf<uint32_t>;
  extern template class ListArrayOf<int64_t>;
#endif

  using ListArray32  = ListArrayOf<int32_t>;
  using ListArrayU32 = ListArrayOf<uint32_t>;
  using ListArray64  = ListArrayOf<int64_t>;
}

#endif // AWKWARD_LISTARRAY_H_

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEXEDARRAY_H_
#define AWKWARD_INDEXEDARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Index.h"
#include "awkward/Content.h"
#include "awkward/kernel-dispatch.h"

namespace awkward {
  /// @class IndexedForm
  ///
  /// @brief Form describing IndexedArray (with `OPTION = false`).
  class LIBAWKWARD_EXPORT_SYMBOL IndexedForm: public Form {
  public:
    /// @brief Creates a IndexedForm. See {@link IndexedArrayOf IndexedArray}
    /// for documentation.
    IndexedForm(bool has_identities,
                const util::Parameters& parameters,
                const FormKey& form_key,
                Index::Form index,
                const FormPtr& content);

    Index::Form
      index() const;

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
    Index::Form index_;
    const FormPtr content_;
  };

  /// @class IndexedOptionForm
  ///
  /// @brief Form describing IndexedOptionArray.
  class LIBAWKWARD_EXPORT_SYMBOL IndexedOptionForm: public Form {
  public:
    /// @brief Creates a IndexedOptionForm. See IndexedArray for documentation.
    IndexedOptionForm(bool has_identities,
                      const util::Parameters& parameters,
                      const FormKey& form_key,
                      Index::Form index,
                      const FormPtr& content);

    Index::Form index() const;

    const FormPtr content() const;

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
    Index::Form index_;
    const FormPtr content_;
  };

  /// @class IndexedArrayOf
  ///
  /// @brief Filters, rearranges, and/or duplicates items in its #content
  /// through an #index, which has the same effect as lazily-applied #carry.
  ///
  /// If `ISOPTION = true`, the array is an IndexedOptionArray with OptionType,
  /// and negative values in the #index correspond to `None`. Otherwise, the
  /// IndexedArray has the same type as its #content.
  ///
  /// See #IndexedArrayOf for the meaning of each parameter.
  template <typename T, bool ISOPTION>
  class
#ifdef AWKWARD_INDEXEDARRAY_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  IndexedArrayOf: public Content {
  public:
    /// @brief Creates an IndexedArray or IndexedOptionArray from a full set
    /// of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param index Item positions to be filtered, rearranged, duplicated, or
    /// masked as `None`.
    /// If #isoption is `true`, negative values are interpreted as `None`.
    /// If #isoption is `false`, negative values are invalid. Values
    /// greater than or equal to `len(content)` are invalid in either case.
    /// @param content Data to be filtered, rearranged, and/or duplicated.
    /// Values in `content[i]` where `i` is not in `index` are
    /// "unreachable;" they do not exist in the high level view.
    IndexedArrayOf<T, ISOPTION>(const IdentitiesPtr& identities,
                                const util::Parameters& parameters,
                                const IndexOf<T>& index,
                                const ContentPtr& content);

    /// @brief Item positions to be filtered, rearranged, duplicated, or
    /// masked as `None`.
    ///
    /// If #isoption is `true`, negative values are interpreted as `None`.
    /// If #isoption is `false`, negative values are invalid. Values
    /// greater than or equal to `len(content)` are invalid in either case.
    const IndexOf<T>
      index() const;

    /// @brief Data to be filtered, rearranged, and/or duplicated.
    ///
    /// Values in `content[i]` where `i` is not in `index` are
    /// "unreachable;" they do not exist in the high level view.
    const ContentPtr
      content() const;

    /// @brief Returns `true` if this array is an IndexedOptionArray32 or
    /// IndexedOptionArray64.
    bool
      isoption() const;

    /// @brief Eagerly applies the #index as a #carry, removing `None`
    /// elements if an IndexedOptionArray.
    const ContentPtr
      project() const;

    /// @brief Performs a set-union of a given `mask` with the missing values
    /// (if an IndexedOptionArray) and calls #project.
    ///
    /// @param mask A byte mask that is valid when `0`, `None` when `1`.
    const ContentPtr
      project(const Index8& mask) const;

    /// @brief Returns an {@link IndexOf Index8} in which each byte represents
    /// missing values with `1` and non-missing values with `0`. The mask
    /// is all `0` if this is an IndexedArray.
    const Index8
      bytemask() const;

    /// @brief If this is an IndexedOptionArray and the #content also has
    /// OptionType, combine the two indicators of missing values into a single
    /// OptionType array. If this is an IndexedArray and the #content is also
    /// an IndexedArray, combine the two #index arrays.
    ///
    /// This is a shallow operation: it only checks the content one level deep.
    const ContentPtr
      simplify_optiontype() const;

    /// @brief Returns the value of #index at a given position.
    T
      index_at_nowrap(int64_t at) const;

    /// @brief User-friendly name of this class: `"IndexedArray32"`,
    /// `"IndexedArrayU32"`, `"IndexedArray64"`,
    /// `"IndexedOptionArray32"`, or `"IndexedOptionArray64"`.
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
    /// Equal to `len(index)`.
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
      getitem_next(const SliceItemPtr& head,
                   const Slice& tail,
                   const Index64& advanced) const override;

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
    /// For {@link IndexedArrayOf IndexedArray} and
    /// {@link IndexedArrayOf IndexedOptionArray}, this method returns
    /// #simplify_optiontype.
    const ContentPtr
      shallow_simplify() const override;

    const ContentPtr
      num(int64_t axis, int64_t depth) const override;

    const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const override;

    bool
      mergeable(const ContentPtr& other, bool mergebool) const override;

    const ContentPtr
      reverse_merge(const ContentPtr& other) const;

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

  protected:
    template <typename S>
    const ContentPtr
      getitem_next_jagged_generic(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const S& slicecontent,
                                  const Slice& tail) const;

    const std::pair<Index64, IndexOf<T>>
      nextcarry_outindex(int64_t& numnull) const;

  private:
    /// @brief See #index.
    const IndexOf<T> index_;
    /// @brief See #content.
    const ContentPtr content_;
  };

#ifndef AWKWARD_INDEXEDARRAY_NO_EXTERN_TEMPLATE
  extern template class IndexedArrayOf<int32_t,  false>;
  extern template class IndexedArrayOf<uint32_t, false>;
  extern template class IndexedArrayOf<int64_t,  false>;
  extern template class IndexedArrayOf<int32_t,  true>;
  extern template class IndexedArrayOf<int64_t,  true>;
#endif

  using IndexedArray32       = IndexedArrayOf<int32_t,  false>;
  using IndexedArrayU32      = IndexedArrayOf<uint32_t, false>;
  using IndexedArray64       = IndexedArrayOf<int64_t,  false>;
  using IndexedOptionArray32 = IndexedArrayOf<int32_t,  true>;
  using IndexedOptionArray64 = IndexedArrayOf<int64_t,  true>;
}

#endif // AWKWARD_INDEXEDARRAY_H_

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONARRAY_H_
#define AWKWARD_UNIONARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Index.h"
#include "awkward/Content.h"
#include "awkward/kernel-dispatch.h"

namespace awkward {
  /// @class UnionForm
  ///
  /// @brief Form describing UnionArray.
  class LIBAWKWARD_EXPORT_SYMBOL UnionForm: public Form {
  public:
    /// @brief Creates a UnionForm. See {@link UnionArrayOf UnionArray} for
    /// documentation.
    UnionForm(bool has_identities,
              const util::Parameters& parameters,
              const FormKey& form_key,
              Index::Form tags,
              Index::Form index,
              const std::vector<FormPtr>& contents);

    Index::Form
      tags() const;

    Index::Form
      index() const;

    const std::vector<FormPtr>
      contents() const;

    int64_t
      numcontents() const;

    const FormPtr
      content(int64_t index) const;

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
    Index::Form tags_;
    Index::Form index_;
    const std::vector<FormPtr> contents_;
  };

  /// @class UnionArrayOf
  ///
  /// @brief Represents heterogeneous data by interleaving several #contents,
  /// indicating which is relevant at a given position with #tags and
  /// where to find each item in the #contents with an interleaved #index.
  ///
  /// See #UnionArrayOf for the meaning of each parameter.
  ///
  /// {@link UnionArrayOf UnionArrays} can be used to interleave data of the
  /// same type, though #simplify_uniontype would combine such arrays to
  /// simplify the representation.
  template <typename T, typename I>
  class
#ifdef AWKWARD_UNIONARRAY_NO_EXTERN_TEMPLATE
  LIBAWKWARD_EXPORT_SYMBOL
#endif
  UnionArrayOf: public Content {
  public:
    /// @brief Generates an index in which `index[i] = i`.
    static const IndexOf<I>
      sparse_index(int64_t len);

    /// @brief Generates an index in which `index[tags == i][i] = i`.
    static const IndexOf<I>
      regular_index(const IndexOf<T>& tags);

    /// @brief Creates a UnionArrayOf from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param tags Small integers indicating which of the #contents to draw
    /// from for each element of the heterogeneous array.
    /// The #tags must all be non-negative and less than `len(contents)`.
    /// @param index Positions within the #contents to find each item.
    /// @param contents `std::vector` of Content instances representing each
    /// of the possible types.
    ///
    /// For each `i`, this `array[i] = contents[tags[i]][index[i]]`.
    UnionArrayOf<T, I>(const IdentitiesPtr& identities,
                       const util::Parameters& parameters,
                       const IndexOf<T> tags,
                       const IndexOf<I>& index,
                       const ContentPtrVec& contents);

    /// @brief Small integers indicating which of the #contents to draw
    /// from for each element of the heterogeneous array.
    ///
    /// The #tags must all be non-negative and less than `len(contents)`.
    ///
    /// For each `i`, this `array[i] = contents[tags[i]][index[i]]`.
    const IndexOf<T>
      tags() const;

    /// @brief Positions within the #contents to find each item.
    ///
    /// For each `i`, this `array[i] = contents[tags[i]][index[i]]`.
    const IndexOf<I>
      index() const;

    /// @brief `std::vector` of Content instances representing each
    /// of the possible types.
    const ContentPtrVec
      contents() const;

    /// @brief The number of #contents.
    int64_t
      numcontents() const;

    /// @brief Returns `contents[index]`.
    const ContentPtr
      content(int64_t index) const;

    /// @brief Returns all items in the array corresponding to one of the
    /// #contents, in the order that they appear in the array.
    ///
    /// Equivalent to `array[tags == index].shallow_simplify()`.
    const ContentPtr
      project(int64_t index) const;

    /// @brief If any of the #contents is also a
    /// {@link UnionArrayOf UnionArray}, combine this array and its #contents
    /// into a single-level {@link UnionArrayOf UnionArray}.
    ///
    /// This is a shallow operation: it only checks the content one level deep.
    const ContentPtr
      simplify_uniontype(bool mergebool) const;

    /// @brief User-friendly name of this class: `"UnionArray8_32"`,
    /// `"UnionArray8_U32"`, or `"UnionArray8_64"`.
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
    /// Equal to `len(tags)`.
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
    /// For {@link UnionArrayOf UnionArray}, this method returns
    /// #simplify_uniontype.
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

  private:
    const IndexOf<T> tags_;
    const IndexOf<I> index_;
    const ContentPtrVec contents_;
  };

#ifndef AWKWARD_UNIONARRAY_NO_EXTERN_TEMPLATE
  extern template class UnionArrayOf<int8_t, int32_t>;
  extern template class UnionArrayOf<int8_t, uint32_t>;
  extern template class UnionArrayOf<int8_t, int64_t>;
#endif

  using UnionArray8_32  = UnionArrayOf<int8_t, int32_t>;
  using UnionArray8_U32 = UnionArrayOf<int8_t, uint32_t>;
  using UnionArray8_64  = UnionArrayOf<int8_t, int64_t>;
}

#endif // AWKWARD_UNIONARRAY_H_

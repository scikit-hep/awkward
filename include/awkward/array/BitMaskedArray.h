// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_BITMASKEDARRAY_H_
#define AWKWARD_BITMASKEDARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Index.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T, bool ISOPTION>
  class IndexedArrayOf;

  class ByteMaskedArray;

  /// @class BitMaskedForm
  ///
  /// @brief Form describing BitMaskedArray.
  class LIBAWKWARD_EXPORT_SYMBOL BitMaskedForm: public Form {
  public:
    /// @brief Creates a BitMaskedForm. See BitMaskedArray for documentation.
    BitMaskedForm(bool has_identities,
                  const util::Parameters& parameters,
                  const FormKey& form_key,
                  Index::Form mask,
                  const FormPtr& content,
                  bool valid_when,
                  bool lsb_order);

    Index::Form
      mask() const;

    const FormPtr
      content() const;

    bool
      valid_when() const;

    bool
      lsb_order() const;

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
    Index::Form mask_;
    const FormPtr content_;
    bool valid_when_;
    bool lsb_order_;
  };

  /// @class BitMaskedArray
  ///
  /// @brief Represents potentially missing data by overlaying a bit #mask
  /// over its #content.
  ///
  /// See #BitMaskedArray for the meaning of each parameter.
  class LIBAWKWARD_EXPORT_SYMBOL BitMaskedArray: public Content {
  public:
    /// @brief Creates an BitMaskedArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param mask Mask in which each bit represents a missing value (`None`)
    /// or a valid value (from #content).
    /// The interpretation depends on #valid_when; only bits that are
    /// equal to #valid_when are not `None`. (Any non-zero value of #valid_when
    /// is equivalent to a `1` bit.)
    /// @param content Data to be masked; `mask[i]` corresponds to `content[i]`
    /// for all `i`.
    /// @param valid_when Interpretation of the boolean bytes in #mask as
    /// `None` or valid values from #content. Only boolean bytes that are
    /// equal to valid_when are not `None`.
    /// @param length Length of the array, since it cannot be determined from
    /// the #mask without an 8-fold ambiguity.
    /// @param lsb_order If `true`, the bits in each byte of the #mask are
    /// taken to be in
    /// [Least Significant Bit (LSB)](https://en.wikipedia.org/wiki/Bit_numbering#LSB_0_bit_numbering)
    /// order; if `false`, they are taken to be in
    /// [Most Significant Bit (MSB)](https://en.wikipedia.org/wiki/Bit_numbering#MSB_0_bit_numbering)
    /// order.
    ///
    /// Any non-zero value of a boolean byte and #valid_when are equivalent.
    BitMaskedArray(const IdentitiesPtr& identities,
                   const util::Parameters& parameters,
                   const IndexU8& mask,
                   const ContentPtr& content,
                   bool valid_when,
                   int64_t length,
                   bool lsb_order);

    /// @brief Mask in which each bit represents a missing value (`None`)
    /// or a valid value (from #content).
    ///
    /// The interpretation depends on #valid_when; only bits that are
    /// equal to #valid_when are not `None`. (Any non-zero value of #valid_when
    /// is equivalent to a `1` bit.)
    const IndexU8
      mask() const;

    /// @brief Data to be masked; `mask[i]` corresponds to `content[i*8]`
    /// through `content[i*8 + 7]` (inclusive) for all `i`.
    const ContentPtr
      content() const;

    /// @brief Interpretation of the boolean bytes in #mask as `None` or
    /// valid values from #content. Only boolean bytes that are equal to
    /// valid_when are not `None`. (Any non-zero value of a boolean byte
    /// and `valid_when` are equivalent.)
    bool
      valid_when() const;

    /// @brief If `true`, the bits in each byte of the #mask are
    /// taken to be in
    /// [Least Significant Bit (LSB)](https://en.wikipedia.org/wiki/Bit_numbering#LSB_0_bit_numbering)
    /// order; if `false`, they are taken to be in
    /// [Most Significant Bit (MSB)](https://en.wikipedia.org/wiki/Bit_numbering#MSB_0_bit_numbering)
    /// order.
    bool
      lsb_order() const;

    /// @brief Return an array with the same type as #content with `None`
    /// values removed.
    const ContentPtr
      project() const;

    /// @brief Performs a set-union of a given `mask` with the missing values
    /// and calls #project.
    ///
    /// @param mask A byte mask that is valid when `0`, `None` when `1`.
    const ContentPtr
      project(const Index8& mask) const;

    /// @brief Expands the #mask to a byte-valued mask with a fixed
    /// interpretation: missing values are `1` and valid values are `0`
    /// (as though #valid_when were `false`).
    const Index8
      bytemask() const;

    /// @brief If the #content also has OptionType, combine the #mask with
    /// the #content's indicator of missing values; also combine if the
    /// #content is a non-OptionType {@link IndexedArrayOf IndexedArray}.
    ///
    /// This is a shallow operation: it only checks the content one level deep.
    const ContentPtr
      simplify_optiontype() const;

    /// @brief Converts this array into a ByteMaskedArray.
    const std::shared_ptr<ByteMaskedArray>
      toByteMaskedArray() const;

    /// @brief Converts this array into an
    /// {@link IndexedArrayOf IndexedOptionArray} with the same missing values.
    const std::shared_ptr<IndexedArrayOf<int64_t, true>>
      toIndexedOptionArray64() const;

    /// @brief User-friendly name of this class: `"BitMaskedArray"`.
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
    /// Note that this is an input parameter.
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
    /// For BitMaskedArray, this method returns #simplify_optiontype.
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

  private:
    /// @brief See #mask.
    const IndexU8 mask_;
    /// @brief See #content.
    const ContentPtr content_;
    /// @brief See #valid_when.
    const bool valid_when_;
    /// @brief See #length.
    const int64_t length_;
    /// @brief See #order.
    const bool lsb_order_;
  };

}

#endif // AWKWARD_BITMASKEDARRAY_H_

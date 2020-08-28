// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_VIRTUALARRAY_H_
#define AWKWARD_VIRTUALARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"
#include "awkward/virtual/ArrayGenerator.h"
#include "awkward/virtual/ArrayCache.h"

namespace awkward {
  /// @class VirtualForm
  ///
  /// @brief Form describing VirtualArray.
  class LIBAWKWARD_EXPORT_SYMBOL VirtualForm: public Form {
  public:
    /// @brief Creates a VirtualForm. See VirtualArray for documentation.
    VirtualForm(bool has_identities,
                const util::Parameters& parameters,
                const FormKey& form_key,
                const FormPtr& form,
                bool has_length);

    bool
      has_form() const;

    const FormPtr
      form() const;

    bool
      has_length() const;

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
    const FormPtr form_;
    bool has_length_;
  };

  /// @class VirtualArray
  ///
  /// @brief Represents an array that can be generated on demand.
  ///
  /// See #VirtualArray for the meaning of each parameter.
  class LIBAWKWARD_EXPORT_SYMBOL VirtualArray: public Content {
  public:
    /// @brief Creates a VirtualArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param generator Function that materializes the array and possibly
    /// checks it against an expected Form.
    /// @param cache Temporary storage for materialized arrays to avoid calling
    /// the #generator more than necessary. May be `nullptr`.
    /// @param cache_key The key this VirtualArray will use when filling a
    /// #cache.
    VirtualArray(const IdentitiesPtr& identities,
                 const util::Parameters& parameters,
                 const ArrayGeneratorPtr& generator,
                 const ArrayCachePtr& cache,
                 const std::string& cache_key,
                 const kernel::lib ptr_lib = kernel::lib::cpu);

    /// @brief Creates a VirtualArray with an automatically assigned #cache_key
    /// (unique per process).
    VirtualArray(const IdentitiesPtr& identities,
                 const util::Parameters& parameters,
                 const ArrayGeneratorPtr& generator,
                 const ArrayCachePtr& cache,
                 const kernel::lib ptr_lib = kernel::lib::cpu);

    /// @brief Function that materializes the array and possibly
    /// checks it against an expected Form.
    const ArrayGeneratorPtr
      generator() const;

    /// @brief Temporary storage for materialized arrays to avoid calling
    /// the #generator more than necessary. May be `nullptr`.
    const ArrayCachePtr
      cache() const;

    const kernel::lib
      ptr_lib() const;

    /// @brief Returns the array if it exists in the #cache; `nullptr`
    /// otherwise.
    ///
    /// This method *does not* cause the array to be materialized if it is not.
    const ContentPtr
      peek_array() const;

    /// @brief Ensures that the array is generated and returns it.
    ///
    /// This method *does not* return `nullptr`.
    const ContentPtr
      array() const;

    /// @brief The key this VirtualArray will use when filling a #cache.
    const std::string
      cache_key() const;

    /// @brief User-friendly name of this class: `"VirtualArray"`.
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

    /// @copydoc Content::nbytes_part
    ///
    /// The bytes of materialized arrays are not counted, so the
    /// {@link Content#nbytes nbytes} of a VirtualArray is always zero.
    void
      nbytes_part(std::map<size_t, int64_t>& largest) const override;

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
      getitem(const Slice& where) const override;

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
      getitem_next(const SliceField& field,
                   const Slice& tail,
                   const Index64& advanced) const override;

    const ContentPtr
      getitem_next(const SliceFields& fields,
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
    /// @brief See #generator.
    const ArrayGeneratorPtr generator_;
    /// @brief See #cache.
    const ArrayCachePtr cache_;
    /// @brief See #cache_key.
    const std::string cache_key_;
    /// @brief See#ptr_lib
    const kernel::lib ptr_lib_;

    /// @brief Forward selected purelist_parameters when making lazy slices
    const util::Parameters
      forward_parameters() const;
  };

}

#endif // AWKWARD_VIRTUALARRAY_H_

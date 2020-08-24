// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NONE_H_
#define AWKWARD_NONE_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class None
  ///
  /// @brief Represents a scalar missing value, which is `None` in Python.
  ///
  /// Nearly all of the methods raise runtime errors. See
  /// {@link Content#isscalar Content::isscalar}.
  class LIBAWKWARD_EXPORT_SYMBOL None: public Content {
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

    /// @exception std::runtime_error is always thrown
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
      carry(const Index64& carry, bool allow_lazy) const override;

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
                  const Index64& shifts,
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

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      localindex(int64_t axis, int64_t depth) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      combinations(int64_t n,
                   bool replacement,
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

    const ContentPtr
      copy_to(kernel::lib ptr_lib) const override;

    const ContentPtr
      numbers_to_type(const std::string& name) const override;

  };

  /// A constant value with type None.
  extern const ContentPtr none;
}

#endif // AWKWARD_NONE_H_

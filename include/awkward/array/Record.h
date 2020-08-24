// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORD_H_
#define AWKWARD_RECORD_H_

#include "awkward/array/RecordArray.h"

namespace awkward {
  /// @class Record
  ///
  /// @brief Represents a tuple or record, a scalar value from RecordArray.
  ///
  /// Many of the methods raise runtime errors. See
  /// {@link Content#isscalar Content::isscalar}.
  class LIBAWKWARD_EXPORT_SYMBOL Record: public Content {
  public:
    /// @brief Creates a Record from a full set of parameters.
    ///
    /// @param array A reference to the array in which this tuple/record
    /// resides (not a copy, shares reference count).
    /// @param at The position in the #array where this tuple/record
    /// resides.
    Record(const std::shared_ptr<const RecordArray> array, int64_t at);

    /// @brief A reference to the array in which this tuple/record
    /// resides (not a copy, shares reference count).
    const std::shared_ptr<const RecordArray>
      array() const;

    /// @brief The position in the #array where this tuple/record
    /// resides.
    int64_t
      at() const;

    /// @brief Returns a `std::vector<std::shared_ptr<Content>>` of each
    /// field at this record position (#at).
    ///
    /// The values might be scalars, such as zero-dimensional NumpyArray,
    /// None, or another Record.
    const ContentPtrVec
      contents() const;

    /// @brief The #array's
    /// {@link RecordArray#recordlookup RecordArray::recordlookup}.
    const util::RecordLookupPtr
      recordlookup() const;

    /// @brief Returns `true` if #recordlookup is `nullptr`; `false` otherwise.
    bool
      istuple() const;

    /// @copydoc Content::isscalar()
    ///
    /// Always returns `true`.
    bool
      isscalar() const override;

    /// @brief User-friendly name of this class: `"Record"`.
    const std::string
      classname() const override;

    const IdentitiesPtr
      identities() const override;

    /// @exception std::runtime_error is always thrown
    void
      setidentities() override;

    /// @exception std::runtime_error is always thrown
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

    const ContentPtr
      getitem_field(const std::string& key) const override;

    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const override;

    /// @exception std::runtime_error is always thrown
    const ContentPtr
      carry(const Index64& carry, bool allow_lazy) const override;

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

    // operations
    const std::string
      validityerror(const std::string& path) const override;

    /// For Record, this method returns #shallow_copy (pass-through).
    const ContentPtr
      shallow_simplify() const override;

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

    /// @brief Returns the field at a given index.
    ///
    /// Equivalent to `contents[fieldindex]`.
    const ContentPtr
      field(int64_t fieldindex) const;

    /// @brief Returns the field at a given key name.
    ///
    /// Equivalent to `contents[fieldindex]`.
    const ContentPtr
      field(const std::string& key) const;

    /// @brief Returns all fields.
    ///
    /// Equivalent to `contents`.
    const ContentPtrVec
      fields() const;

    /// @brief Returns key, field pairs for all fields.
    const std::vector<std::pair<std::string, ContentPtr>>
      fielditems() const;

    /// @brief Returns this Record without #recordlookup, converting any
    /// records into tuples.
    const std::shared_ptr<Record>
      astuple() const;

    const ContentPtr
      getitem(const Slice& where) const override;

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

  private:
    /// @brief See #array.
    const std::shared_ptr<const RecordArray> array_;
    /// @brief See #at.
    int64_t at_;
  };
}

#endif // AWKWARD_RECORD_H_

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDARRAY_H_
#define AWKWARD_RECORDARRAY_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "awkward/common.h"
#include "awkward/Identities.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class RecordForm
  ///
  /// @brief Form describing RecordArray (not a Record).
  class LIBAWKWARD_EXPORT_SYMBOL RecordForm: public Form {
  public:
    /// @brief Creates a RecordForm. See RecordArray (not Record) for
    /// documentation.
    RecordForm(bool has_identities,
               const util::Parameters& parameters,
               const FormKey& form_key,
               const util::RecordLookupPtr& recordlookup,
               const std::vector<FormPtr>& contents);

    const util::RecordLookupPtr
      recordlookup() const;

    const std::vector<FormPtr>
      contents() const;

    bool
      istuple() const;

    const FormPtr
      content(int64_t fieldindex) const;

    const FormPtr
      content(const std::string& key) const;

    const std::vector<std::pair<std::string, FormPtr>>
      items() const;

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
    const util::RecordLookupPtr recordlookup_;
    const std::vector<FormPtr> contents_;
  };

  /// @class RecordArray
  ///
  /// @brief Represents an array of tuples or records, in which a tuple
  /// has a fixed number of differently typed fields and a record has
  /// a named set of differently typed fields.
  ///
  /// See #RecordArray for the meaning of each parameter.
  ///
  /// Tuples and records are distinguished by the absence or presence of a
  /// #recordlookup (`std::vector<std::string>`) that associates key names
  /// with each field.
  ///
  /// Fields are always ordered, whether tuples or records.
  class LIBAWKWARD_EXPORT_SYMBOL RecordArray:
    public Content,
    public std::enable_shared_from_this<RecordArray> {
  public:
    /// @brief Creates a RecordArray from a full set of parameters.
    ///
    /// @param identities Optional Identities for each element of the array
    /// (may be `nullptr`).
    /// @param parameters String-to-JSON map that augments the meaning of this
    /// array.
    /// @param contents `std::vector` of Content instances representing the
    /// (ordered) fields.
    /// @param recordlookup A `std::shared_ptr<std::vector<std::string>>`
    /// optional list of key names.
    /// If absent (`nullptr`), the data are tuples; otherwise, they are
    /// records. The number of names must match the number of #contents.
    /// @param length The length of the array, breaking ambiguities between
    /// #contents of different lengths and essential if there are zero fields.
    RecordArray(const IdentitiesPtr& identities,
                const util::Parameters& parameters,
                const ContentPtrVec& contents,
                const util::RecordLookupPtr& recordlookup,
                int64_t length);

    /// @brief Creates a RecordArray in which #length is the minimum
    /// length of the #contents or zero if there are no #contents.
    RecordArray(const IdentitiesPtr& identities,
                const util::Parameters& parameters,
                const ContentPtrVec& contents,
                const util::RecordLookupPtr& recordlookup);

    /// @brief `std::vector` of Content instances representing the
    /// (ordered) fields.
    const ContentPtrVec
      contents() const;

    /// @brief A `std::shared_ptr<std::vector<std::string>>`
    /// optional list of key names.
    /// If absent (`nullptr`), the data are tuples; otherwise, they are
    /// records. The number of names must match the number of #contents.
    const util::RecordLookupPtr
      recordlookup() const;

    /// @brief Returns `true` if #recordlookup is `nullptr`; `false` otherwise.
    bool
      istuple() const;

    /// @brief Returns a RecordArray with an additional or a replaced field
    /// at index `where` with value `what`.
    ///
    /// This "setitem" method does not change the original array; and the
    /// output references most of the original data (without copying).
    const ContentPtr
      setitem_field(int64_t where, const ContentPtr& what) const;

    /// @brief Returns a RecordArray with an additional or a replaced field
    /// at key name `where` with value `what`.
    ///
    /// This "setitem" method does not change the original array; and the
    /// output references most of the original data (without copying).
    const ContentPtr
      setitem_field(const std::string& where, const ContentPtr& what) const;

    /// @brief User-friendly name of this class: `"RecordArray"`.
    const std::string
      classname() const override;

    void
      setidentities() override;

    void
      setidentities(const IdentitiesPtr& identities) override;

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

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

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
    /// For RecordArray, this method returns #shallow_copy (pass-through).
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

    /// @brief Returns the field at a given index (without trimming it to
    /// have the same #length as this RecordArray).
    ///
    /// Equivalent to `contents[fieldindex]`.
    const ContentPtr
      field(int64_t fieldindex) const;

    /// @brief Returns the field with a given key name (without trimming it to
    /// have the same #length as this RecordArray).
    ///
    /// Equivalent to `contents[fieldindex(key)]`.
    const ContentPtr
      field(const std::string& key) const;

    /// @brief Returns all the fields (without trimming them to have the same
    /// #length as this RecordArray).
    ///
    /// Equivalent to `contents`.
    const ContentPtrVec
      fields() const;

    /// @brief Returns key, field pairs for all fields (without trimming them
    /// to have the same #length as this RecordArray).
    const std::vector<std::pair<std::string, ContentPtr>>
      fielditems() const;

    /// @brief Returns this RecordArray without #recordlookup, converting any
    /// records into tuples.
    const std::shared_ptr<RecordArray>
      astuple() const;

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

  protected:
    template <typename S>
    const ContentPtr
      getitem_next_jagged_generic(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const S& slicecontent,
                                  const Slice& tail) const;

  private:
    /// @brief See #contents.
    const ContentPtrVec contents_;
    /// @brief See #recordlookup.
    const util::RecordLookupPtr recordlookup_;
    /// @brief See #length.
    int64_t length_;
  };
}

#endif // AWKWARD_RECORDARRAY_H_

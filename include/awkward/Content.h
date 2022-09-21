// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include <map>

#include "awkward/common.h"
#include "awkward/Identities.h"
#include "awkward/Slice.h"
#include "awkward/Index.h"

namespace awkward {
  class Content;
  using ContentPtr    = std::shared_ptr<Content>;
  using ContentPtrVec = std::vector<std::shared_ptr<Content>>;
  class Form;
  using FormPtr       = std::shared_ptr<Form>;
  using FormKey       = std::shared_ptr<std::string>;
  class ArrayCache;
  using ArrayCachePtr = std::shared_ptr<ArrayCache>;
  class Type;
  using TypePtr = std::shared_ptr<Type>;
  class Reducer;
  class ToJson;
  class ToJsonPrettyString;
  class ToJsonString;
  class ToJsonPrettyFile;
  class ToJsonFile;

  /// @class Form
  ///
  /// @brief Abstract superclass of all array node forms, which expresses the
  /// nesting structure without any large {@link IndexOf Index} or data
  /// buffers.
  ///
  /// Forms may be thought of as low-level types, whereas Type is a high-level
  /// type. There is a one-to-many relationship from Type to Form.
  class LIBAWKWARD_EXPORT_SYMBOL Form {
  public:
    static FormPtr
      fromnumpy(char kind,
                int64_t itemsize,
                const std::vector<int64_t>& inner_shape);
    static FormPtr
      fromjson(const std::string& data);

    /// @brief Called by subclass constructors; assigns #has_identities,
    /// #parameters, and #form_key upon construction.
    Form(bool has_identities,
         const util::Parameters& parameters,
         const FormKey& form_key);

    /// @brief Empty destructor; required for some C++ reason.
    virtual ~Form() { }

    /// @brief High-level Type describing this Form.
    ///
    /// @param typestrs A mapping from `"__record__"` parameters to string
    /// representations of those types, to override the derived strings.
    virtual const TypePtr
      type(const util::TypeStrs& typestrs) const = 0;

    /// @brief Internal function to produce a JSON representation one node at
    /// a time.
    virtual void
      tojson_part(ToJson& builder, bool verbose) const = 0;

    /// @brief Copies this node without copying any nodes hierarchically
    /// nested within it.
    virtual const FormPtr
      shallow_copy() const = 0;

    /// @brief Copies this node, adding or replacing a form_key.
    virtual const FormPtr
      with_form_key(const FormKey& form_key) const = 0;

    /// @brief Returns `true` if this Form is equal to the other Form; `false`
    /// otherwise.
    ///
    /// @param check_identities If `true`, Forms are not equal unless they both
    /// #has_identities.
    /// @param check_parameters If `true`, Forms are not equal unless they have
    /// the same #parameters.
    /// @param check_form_key If `true`, Forms are not equal unless they have
    /// the same #form_key.
    /// @param compatibility_check If `true`, this is part of a compatibility
    /// check between an expected Form (`this`) and a generated array's Form
    /// (`other`). When the expected Form is a VirtualForm, it's allowed to be
    /// less specific than the `other` VirtualForm.
    virtual bool
      equal(const FormPtr& other,
            bool check_identities,
            bool check_parameters,
            bool check_form_key,
            bool compatibility_check) const = 0;

    /// @brief Returns `true` if this Form has the same #form_key as the other.
    bool
      form_key_equals(const FormKey& other_form_key) const;

    /// @brief The parameter associated with `key` at the first level
    /// that has a non-null value, descending only as deep as the first
    /// RecordForm.
    virtual const std::string
      purelist_parameter(const std::string& key) const = 0;

    /// @brief Returns `true` if the parameter associated with `key` exists
    /// and is equal to `value`; `false` otherwise.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// Equality is checked at the level of JSON DOMs. The `value` does not
    /// need to be exactly the same string; it needs to have equivalent JSON
    /// value.
    bool
      parameter_equals(const std::string& key, const std::string& value) const;

    /// @brief Returns `true` if all nested lists down to the first RecordForm
    /// are RegularForm nodes; `false` otherwise.
    virtual bool
      purelist_isregular() const = 0;

    /// @brief The list-depth of this array, not counting any contained
    /// within a RecordForm.
    ///
    /// If this array contains a UnionForm with different depths, the return
    /// value is `-1`.
    virtual int64_t
      purelist_depth() const = 0;

    /// @brief Returns `true` if this dimension has option-type; `false`
    /// otherwise.
    virtual bool
      dimension_optiontype() const = 0;

    /// @brief An optional string associated with this Form, usually specifying
    /// where an array may be fetched.
    const FormKey
      form_key() const;

    /// @brief Returns (a) the minimum list-depth and (b) the maximum
    /// list-depth of the array, which can differ if this array "branches"
    /// (differs when followed through different fields of a RecordForm or
    /// UnionForm).
    virtual const std::pair<int64_t, int64_t>
      minmax_depth() const = 0;

    /// @brief Returns (a) whether the list-depth of this array "branches,"
    /// or differs when followed through different fields of a RecordForm or
    /// UnionForm and (b) the minimum list-depth.
    ///
    /// If the array does not contain any records or heterogeneous data, the
    /// `first` element is always `true` and the `second` is simply the depth.
    virtual const std::pair<bool, int64_t>
      branch_depth() const = 0;

    /// @brief The number of fields in the first nested tuple or
    /// records or `-1` if this array does not contain a RecordForm.
    virtual int64_t
      numfields() const = 0;

    /// @brief The position of a tuple or record key name if this array
    /// contains a RecordForm.
    virtual int64_t
      fieldindex(const std::string& key) const = 0;

    /// @brief The record name associated with a given field index or
    /// the tuple index as a string (e.g. `"0"`, `"1"`, `"2"`) if a tuple.
    ///
    /// Raises an error if the array does not contain a RecordForm.
    virtual const std::string
      key(int64_t fieldindex) const = 0;

    /// @brief Returns `true` if the array contains a RecordForm with the
    /// specified `key`; `false` otherwise.
    virtual bool
      haskey(const std::string& key) const = 0;

    /// @brief A list of RecordArray keys or an empty list if this
    /// array does not contain a RecordArray.
    virtual const std::vector<std::string>
      keys() const = 0;

    /// @brief Returns `true` if the outermost RecordArray is a tuple
    virtual bool
      istuple() const = 0;

    /// @brief Returns a string representation of this Form (#tojson with
    /// `pretty = true` and `verbose = false`).
    virtual const std::string
      tostring() const;

    /// @brief Returns a JSON representation of this array.
    ///
    /// @param pretty If `true`, add spacing to make the JSON human-readable.
    /// If `false`, return a compact representation.
    virtual const std::string
      tojson(bool pretty, bool verbose) const;

    /// @brief Returns `true` if the corresponding array has associated
    /// {@link IdentitiesOf Identities}.
    bool
      has_identities() const;

    /// @brief String-to-JSON map that augments the meaning of this
    /// Form.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const util::Parameters
      parameters() const;

    /// @brief Get one parameter from this Form.
    ///
    /// If the `key` does not exist, this function returns `"null"`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const std::string
      parameter(const std::string& key) const;

    /// @brief Internal function for adding identities in #tojson.
    ///
    /// Must be called between `builder.beginrecord()` and `builder.endrecord()`.
    void
      identities_tojson(ToJson& builder, bool verbose) const;

    /// @brief Internal function for adding parameters in #tojson.
    ///
    /// Must be called between `builder.beginrecord()` and `builder.endrecord()`.
    void
      parameters_tojson(ToJson& builder, bool verbose) const;

    /// @brief Internal function for adding form_key in #tojson.
    ///
    /// Must be called between `builder.beginrecord()` and `builder.endrecord()`.
    void
      form_key_tojson(ToJson& builder, bool verbose) const;

    /// @brief Returns the Form that would result from a range-slice.
    ///
    /// Matches the operation of Content#getitem_range.
    virtual const FormPtr
      getitem_range() const;

    /// @brief Returns the Form that would result from a field-slice.
    ///
    /// Matches the operation of Content#getitem_field.
    virtual const FormPtr
      getitem_field(const std::string& key) const = 0;

    /// @brief Returns the Form that would result from a fields-slice.
    ///
    /// Matches the operation of Content#getitem_fields.
    virtual const FormPtr
      getitem_fields(const std::vector<std::string>& keys) const = 0;

  protected:
    /// @brief See #has_identities
    bool has_identities_;
    /// @brief See #parameters
    util::Parameters parameters_;
    /// @brief See #form_key
    FormKey form_key_;
  };

  /// @class Content
  ///
  /// @brief Abstract superclass of all array node types (flat hierarchy).
  /// Any Content can be nested within any other Content.
  class LIBAWKWARD_EXPORT_SYMBOL Content {
  };
}

#endif // AWKWARD_CONTENT_H_

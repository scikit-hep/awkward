// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include <cstdio>
#include <map>

#include "awkward/common.h"
#include "awkward/Identities.h"
#include "awkward/Slice.h"
#include "awkward/io/json.h"
#include "awkward/type/Type.h"
#include "awkward/Index.h"
#include "awkward/Reducer.h"

namespace awkward {
  class Content;
  using ContentPtr    = std::shared_ptr<Content>;
  using ContentPtrVec = std::vector<std::shared_ptr<Content>>;
  class Form;
  using FormPtr       = std::shared_ptr<Form>;
  using FormKey       = std::shared_ptr<std::string>;

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

    /// @brief Internal function for extracting record field
    ///
    /// WARNING: this function returns the field from the innermost record
    /// form it finds, and does not wrap it with the context.
    /// Used by VirtualArray::getitem_field to determine certain parameters
    /// without materialization. A possible extension would be to wrap output
    /// at each layer to fully specify the form.
    virtual const FormPtr
      getitem_field(const std::string& key) const = 0;

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
  public:
    /// @brief Called by all subclass constructors; assigns #identities and
    /// #parameters upon construction.
    Content(const IdentitiesPtr& identities,
            const util::Parameters& parameters);

    /// @brief Empty destructor; required for some C++ reason.
    virtual ~Content() { }

    /// @brief Returns `true` if the data represented by this node is scalar,
    /// not a true array.
    ///
    /// Scalars are not valid inputs to many operations, but they are
    /// instances of Content so that they can be outputs of some operations.
    ///
    /// Formally, scalars and arrays should probably be distinct C++ types
    /// to eliminate this ambiguity, but that would introduce excessive
    /// type-testing and casting in all the algorithms that chain operations
    /// (output of one is input to another). It is more convenient to test
    /// for scalars at runtime with this method.
    ///
    /// The only scalars are:
    ///
    ///    - NumpyArray with `shape.empty()` (zero-dimensional array), which
    ///      represents scalar numbers and booleans. (Immediately converted
    ///      to a number object in Python.)
    ///    - None, which represents missing values. (Immediately converted
    ///      to `None` in Python.)
    ///    - Record, which represents one item from a RecordArray.
    virtual bool
      isscalar() const;

    /// @brief User-friendly name of this class, including integer-type
    /// specialization.
    virtual const std::string
      classname() const = 0;

    /// @brief Optional Identities for each element of the array
    /// (may be `nullptr`).
    virtual const IdentitiesPtr
      identities() const;

    /// @brief Assign a surrogate index of Identities to this array (in-place).
    ///
    /// This also assigns and possibly replaces Identities in nested arrays.
    ///
    /// @note This mutability is temporary:
    /// [scikit-hep/awkward-1.0#116](https://github.com/scikit-hep/awkward-1.0/issues/176)
    /// Eventually, this interface will be deprecated and all Content
    /// instances will be immutable.
    virtual void
      setidentities() = 0;

    /// @brief Assign a specified set of Identities to this array (in-place).
    ///
    /// This also assigns and possibly replaces Identities in nested arrays.
    ///
    /// @note This mutability is temporary:
    /// [scikit-hep/awkward-1.0#116](https://github.com/scikit-hep/awkward-1.0/issues/176)
    /// Eventually, this interface will be deprecated and all Content
    /// instances will be immutable.
    virtual void
      setidentities(const IdentitiesPtr& identities) = 0;

    /// @brief High-level Type describing this array.
    ///
    /// @param typestrs A mapping from `"__record__"` parameters to string
    /// representations of those types, to override the derived strings.
    virtual const TypePtr
      type(const util::TypeStrs& typestrs) const = 0;

    /// @brief Low-level Form describing all the features of this array
    /// except the actual data buffers ({@link IndexOf Index},
    /// {@link IdentitiesOf Identities},
    /// {@link NumpyArray#ptr NumpyArray::ptr}, and
    /// {@link RawArrayOf#ptr RawArray::ptr}).
    ///
    /// @param materialize If true, materialize any
    /// {@link VirtualArray VirtualArrays} encountered along the way.
    virtual const FormPtr
      form(bool materialize) const = 0;

    /// @brief If true, the array contains a VirtualArray with an unknown Form.
    virtual bool
      has_virtual_form() const = 0;

    /// @brief If true, the array contains a VirtualArray with an unknown
    /// length.
    virtual bool
      has_virtual_length() const = 0;

    /// @brief Internal function to build an output string for #tostring.
    ///
    /// @param indent Indentation depth as a string of spaces.
    /// @param pre Prefix string, usually an opening XML tag.
    /// @param post Postfix string, usually a closing XML tag and carriage
    /// return.
    virtual const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;

    /// @brief Internal function to produce a JSON representation one node at
    /// a time.
    virtual void
      tojson_part(ToJson& builder, bool include_beginendlist) const = 0;

    /// @brief Internal function used to calculate #nbytes.
    ///
    /// @param largest The largest range of bytes used in each
    /// reference-counted pointer (`size_t`).
    ///
    /// @note This method of accounting for overlapping buffers is
    /// insufficient: two nodes could use parts of the same buffer in which
    /// one doesn't completely overlap the other. It's not likely, but
    /// currently the calculation of #nbytes is an underestimate.
    virtual void
      nbytes_part(std::map<size_t, int64_t>& largest) const = 0;

    /// @brief The number of elements in the array.
    virtual int64_t
      length() const = 0;

    /// @brief Copies this node without copying any nodes hierarchically
    /// nested within it or any array/index/identity buffers.
    ///
    /// See also #deep_copy.
    virtual const ContentPtr
      shallow_copy() const = 0;

    /// @brief Copies this node and all nodes hierarchically
    /// nested within it, optionally copying the associated arrays, indexes,
    /// and identities, too.
    ///
    /// See also #shallow_copy.
    ///
    /// @param copyarrays If `true`, copy the associated array buffers (in
    /// NumpyArray and {@link RawArrayOf RawArray}), not just the lightweight
    /// objects that point to them.
    /// @param copyindexes If `true`, copy the {@link IndexOf Index} objects
    /// and their buffers as well.
    /// @param copyidentities If `true`, copy the
    /// {@link IdentitiesOf Identities} objects and their buffers as well.
    virtual const ContentPtr
      deep_copy(bool copyarrays,
                bool copyindexes,
                bool copyidentities) const = 0;

    /// @brief Performs up-front validity checks on an array so that they don't
    /// have to be checked in #getitem_at_nowrap for each item.
    virtual void
      check_for_iteration() const = 0;

    /// @brief Internal function to get an empty slice (with the correct
    /// type).
    virtual const ContentPtr
      getitem_nothing() const = 0;

    /// @brief Returns the element at a given position in the array, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    virtual const ContentPtr
      getitem_at(int64_t at) const = 0;

    /// @brief Returns the element at a given position in the array, without
    /// handling negative indexing or bounds-checking.
    ///
    /// If the array has Identities, the identity bounds are checked.
    virtual const ContentPtr
      getitem_at_nowrap(int64_t at) const = 0;

    /// @brief Subinterval of this array, handling negative indexing
    /// and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    ///
    /// Ranges beyond the array are not an error; they are trimmed to
    /// `start = 0` on the left and `stop = length() - 1` on the right.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    virtual const ContentPtr
      getitem_range(int64_t start, int64_t stop) const = 0;

    /// @brief Subinterval of this array, without handling negative
    /// indexing or bounds-checking.
    ///
    /// If the array has Identities, the identity bounds are checked.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    virtual const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const = 0;

    /// @brief This array with the first nested RecordArray replaced by
    /// the field at `key`.
    virtual const ContentPtr
      getitem_field(const std::string& key) const = 0;

    /// @brief This array with the first nested RecordArray replaced by
    /// a RecordArray of a given subset of `keys`.
    virtual const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const = 0;

    /// @brief Entry point for general slicing: Slice represents a tuple of
    /// SliceItem nodes applying to each level of nested lists.
    virtual const ContentPtr
      getitem(const Slice& where) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// @param head First element of the Slice tuple.
    /// @param tail The rest of the elements of the Slice tuple.
    /// @param advanced If empty, no array slices (integer or boolean) have
    /// been encountered yet; otherwise, positions in any subsequent array
    /// slices to select.
    ///
    /// In the [NumPy documentation](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing),
    /// advanced indexes are described as iterating "as one," which requires
    /// an {@link IndexOf Index} to be propagated when implemented recursively.
    virtual const ContentPtr
      getitem_next(const SliceItemPtr& head,
                   const Slice& tail,
                   const Index64& advanced) const;

    /// @brief Internal function that propagates a jagged array (array with
    /// irregular-length dimensions) slice through one axis.
    ///
    /// @param slicestarts Effective `starts` (similar to
    /// {@link ListArrayOf ListArray}'s `starts`) of the jagged slice.
    /// @param slicestops Effective `stops` of the jagged slice.
    /// @param slicecontent Nested `content` within the jagged slice.
    /// @param tail Subsequent SliceItem elements beyond the jagged array
    /// hierarchy.
    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceItemPtr& slicecontent,
                          const Slice& tail) const;

    /// @brief Returns an array of the same type with elements filtered,
    /// rearranged, and possibly duplicated by the `carry` array of integers.
    ///
    /// The output has the same length as the `carry` index, not the `array`
    /// that is being manipulated. For each item `i` in `carry`, the output
    /// is `array[index[i]]`.
    ///
    /// This operation is called
    /// [take](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html)
    /// in NumPy and Arrow, although this #carry is a low-level function that
    /// does not handle negative indexes and is not exposed to the Python
    /// layer. It is used by many operations to pass
    /// filters/rearrangements/duplications from one typed array node to
    /// another without knowing the latter's type.
    ///
    /// Taking #getitem_at_nowrap as a function from integers to the array's
    /// item type, `A: [0, len(a)) → T`, and the `carry` array's
    /// {@link IndexOf#getitem_at_nowrap Index64::getitem_at_nowrap} as a
    /// function `C: [0, len(c)) → [0, len(a))`, this method represents
    /// function composition, `A ∘ C: [0, len(c)) → T`.
    ///
    /// @note If the `carry` array has duplicate elements, the array is only
    /// duplicated one level deep. For instance, on a
    /// {@link ListArrayOf ListArray}, only the `starts` and `stops` get
    /// duplicated, not the `content` (and similarly for all other node types).
    virtual const ContentPtr
      carry(const Index64& carry, bool allow_lazy) const = 0;

    /// @brief The parameter associated with `key` at the first level
    /// that has a non-null value, descending only as deep as the first
    /// RecordArray.
    virtual const std::string
      purelist_parameter(const std::string& key) const;

    /// @brief Returns `true` if all nested lists down to the first RecordArray
    /// are RegularArray nodes; `false` otherwise.
    bool
      purelist_isregular() const;

    /// @brief The list-depth of this array, not counting any contained
    /// within a RecordArray.
    ///
    /// The `purelist_depth` of a Record is `0`, and a RecordArray is `1`
    /// (regardless of what its fields contain).
    ///
    /// If this array contains a {@link UnionArrayOf UnionArray} with
    /// different depths, the return value is `-1`.
    virtual int64_t
      purelist_depth() const;

    /// @brief Returns (a) the minimum list-depth and (b) the maximum
    /// list-depth of the array, which can differ if this array "branches"
    /// (differs when followed through different fields of a RecordArray or
    /// {@link UnionArrayOf UnionArray}).
    virtual const std::pair<int64_t, int64_t>
      minmax_depth() const;

    /// @brief Returns (a) whether the list-depth of this array "branches,"
    /// or differs when followed through different fields of a RecordArray or
    /// {@link UnionArrayOf UnionArray} and (b) the minimum list-depth.
    ///
    /// If the array does not contain any records or heterogeneous data, the
    /// `first` element is always `true` and the `second` is simply the depth.
    virtual const std::pair<bool, int64_t>
      branch_depth() const;

    /// @brief The number of fields in the first nested tuple or
    /// records or `-1` if this array does not contain a RecordArray.
    virtual int64_t
      numfields() const = 0;

    /// @brief The position of a tuple or record key name if this array
    /// contains a RecordArray.
    virtual int64_t
      fieldindex(const std::string& key) const = 0;

    /// @brief The record name associated with a given field index or
    /// the tuple index as a string (e.g. `"0"`, `"1"`, `"2"`) if a tuple.
    ///
    /// Raises an error if the array does not contain a RecordArray.
    virtual const std::string
      key(int64_t fieldindex) const = 0;

    /// @brief Returns `true` if the array contains a RecordArray with the
    /// specified `key`; `false` otherwise.
    virtual bool
      haskey(const std::string& key) const = 0;

    /// @brief A list of RecordArray keys or an empty list if this
    /// array does not contain a RecordArray.
    virtual const std::vector<std::string>
      keys() const = 0;

    /// @brief Returns an error message if this array is invalid; otherwise,
    /// returns an empty string.
    virtual const std::string
      validityerror(const std::string& path) const = 0;

    /// @brief Returns an equivalent array simplified at one level only using
    /// {@link IndexedArrayOf#simplify_optiontype simplify_optiontype}
    /// if an option-type array and
    /// {@link UnionArrayOf#simplify_uniontype simplify_uniontype}
    /// if a union-type array.
    ///
    /// For all other types of arrays, this operation is a pass-through.
    virtual const ContentPtr
      shallow_simplify() const = 0;

    /// @brief The length of this array (as a NumpyArray scalar) if `axis = 0`
    /// or the lengths of subarrays (as an array or nested array) if
    /// `axis != 0`.
    ///
    /// @param axis The axis whose length or lengths to quantify.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param depth The current depth while stepping into the array: this
    /// value is set to `0` on the array node where the user starts the
    /// process and is increased at each level of list-depth (instead of
    /// decreasing the user-specified `axis`).
    virtual const ContentPtr
      num(int64_t axis, int64_t depth) const = 0;

    /// @brief Returns (a) an offsets {@link IndexOf Index} and (b) a flattened
    /// version of the array at some `axis` depth.
    ///
    /// If `axis > 1` (or its negative equivalent), the offsets is empty.
    ///
    /// @param axis The axis to eliminate by flattening. `axis = 0` is
    /// invalid.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param depth The current depth while stepping into the array: this
    /// value is set to `0` on the array node where the user starts the
    /// process and is increased at each level of list-depth (instead of
    /// decreasing the user-specified `axis`).
    virtual const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const = 0;

    /// @brief Returns `true` if this array can be merged with the `other`;
    /// `false` otherwise.
    ///
    /// The #merge method will complete without errors if this function
    /// returns `true`.
    ///
    /// @param other The other array to merge with.
    /// @param mergebool If `true`, consider boolean types to be equivalent
    /// to integers.
    virtual bool
      mergeable(const ContentPtr& other, bool mergebool) const = 0;

    /// @brief An array with this and the `other` concatenated (this
    /// first, `other` last).
    virtual const ContentPtr
      merge(const ContentPtr& other) const = 0;

    /// @brief Converts this array into a SliceItem that can be used in
    /// getitem.
    virtual const SliceItemPtr
      asslice() const = 0;

    /// @brief Returns this array with `None` values replaced by a given
    /// `value`.
    ///
    /// @param value An array of exactly one element, which need not have
    /// the same type as the missing values it's replacing.
    virtual const ContentPtr
      fillna(const ContentPtr& value) const = 0;

    /// @brief If `axis = 0`, returns a view of this array padded on the
    /// right with `None` values to have a minimum length; otherwise, returns
    /// an array with nested lists all padded to the minimum length.
    ///
    /// @param target The intended length. The output may be longer than this
    /// target length, but not shorter (using {@link ListArrayOf ListArray}).
    /// @param axis The axis at which to apply padding.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param depth The current depth while stepping into the array: this
    /// value is set to `0` on the array node where the user starts the
    /// process and is increased at each level of list-depth (instead of
    /// decreasing the user-specified `axis`).
    virtual const ContentPtr
      rpad(int64_t target, int64_t axis, int64_t depth) const = 0;

    /// @brief If `axis = 0`, returns a view of this array padded on the right
    //// with `None` values to have exactly the specified length; otherwise,
    /// returns an array with nested lists all padded to the specified length.
    ///
    /// @param target The intended length; the output has exactly this target
    /// length (using RegularArray).
    /// @param axis The axis at which to apply padding.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param depth The current depth while stepping into the array: this
    /// value is set to `0` on the array node where the user starts the
    /// process and is increased at each level of list-depth (instead of
    /// decreasing the user-specified `axis`).
    virtual const ContentPtr
      rpad_and_clip(int64_t target, int64_t axis, int64_t depth) const = 0;

    /// @brief This array with one axis removed by applying a Reducer
    /// (e.g. "sum", "max", "any", "all).
    ///
    /// The user's entry point for this operation is #reduce.
    ///
    /// @param reducer The choice of Reducer algorithm.
    /// @param negaxis The negative axis: `-axis`. That is, `negaxis = 1`
    /// means the deepest axis level.
    /// @param starts Staring positions of each group to combine as an
    /// {@link IndexOf Index}. These are downward pointers from an outer
    /// structure into this structure with the same meaning as in
    /// {@link ListArrayOf ListArray}. Only used by a reducer that
    /// {@link Reducer#returns_positions returns_positions} (currently
    /// only argmin and argmax).
    /// @param shifts Per-element adjustments for any reducer that
    /// {@link Reducer#returns_positions returns_positions}. Allows
    /// for variable-length lists with axis != -1 and for missing values
    /// (None) in argmin and argmax.
    /// @param parents Groups to combine as an {@link IndexOf Index} of
    /// upward pointers from this structure to the outer structure to reduce.
    /// @param outlength The length of the reduced array, after the operation
    /// completes.
    /// @param mask If `true`, the Reducer's identity values will be covered
    /// by `None` using a ByteMaskedArray. This is desirable for ReducerMin,
    /// ReducerMax, ReducerArgmin, and ReducerArgmax to indicate that empty
    /// lists have no minimum or maximum.
    /// @param keepdims If `true`, the reduced values will be wrapped by a
    /// singleton RegularArray to maintain the same number of dimensions in
    /// the output.
    virtual const ContentPtr
      reduce_next(const Reducer& reducer,
                  int64_t negaxis,
                  const Index64& starts,
                  const Index64& shifts,
                  const Index64& parents,
                  int64_t outlength,
                  bool mask,
                  bool keepdims) const = 0;

    /// @brief This array with one axis removed by applying a Reducer
    ///
    /// The user's entry point for this operation is #sort.
    ///
    /// @param negaxis The negative axis: `-axis`. That is, `negaxis = 1`
    /// means the deepest axis level.
    /// @param starts Staring positions of each group to combine as an
    /// {@link IndexOf Index}. These are downward pointers from an outer
    /// structure into this structure with the same meaning as in
    /// {@link ListArrayOf ListArray}.
    /// @param parents Groups to combine as an {@link IndexOf Index} of
    /// upward pointers from this structure to the outer structure to reduce.
    /// @param outlength The length of the array, after the operation
    /// completes.
    /// @param ascending If `true`, the values will be sorted in an ascending
    /// order.
    /// @param stable If `true`, the values will be sorted by a
    /// stable sort algorithm to maintain the relative order of records with
    /// equal keys (i.e. values).
    virtual const ContentPtr
      sort_next(int64_t negaxis,
                const Index64& starts,
                const Index64& parents,
                int64_t outlength,
                bool ascending,
                bool stable,
                bool keepdims) const = 0;

    virtual const ContentPtr
      argsort_next(int64_t negaxis,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength,
                   bool ascending,
                   bool stable,
                   bool keepdims) const = 0;

    /// @brief A (possibly nested) array of integers indicating the
    /// positions of elements within each nested list.
    ///
    /// @param axis The nesting depth at which this operation is applied.
    /// If `axis = 0`, the output is simply an array of integers from `0`
    /// to `length()`. If `axis = 1`, the output has one level of nesting
    /// containing integers from `0` to the length of the nested list.
    /// Higher values of `axis` leave outer layers of the structure untouched.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param depth The current depth while stepping into the array: this
    /// value is set to `0` on the array node where the user starts the
    /// process and is increased at each level of list-depth (instead of
    /// decreasing the user-specified `axis`).
    virtual const ContentPtr
      localindex(int64_t axis, int64_t depth) const = 0;

    /// @brief Tuples or records of all `n`-tuple combinations of
    /// list items at some `axis` depth.
    ///
    /// For example, the `n = 2` combinations at `axis = 0` of
    ///
    /// @code{.py}
    /// [a, b, c, d, e]
    /// @endcode
    ///
    /// would be
    ///
    /// @code{.py}
    /// [[(a, b), (a, c), (a, d), (a, e)],
    ///  [(b, c), (b, d), (b, e)],
    ///  [(c, d), (c, e)],
    ///  [(d, e)]]
    /// @endcode
    ///
    /// and the `n = 3` combinations at `axis = 1` of
    ///
    /// @code{.py}
    /// [[a, b, c, d], [], [e, f], [g, h, i]]
    /// @endcode
    ///
    /// would be
    ///
    /// @code{.py}
    /// [[[(a, b, c), (a, b, d), (a, c, d)], [(b, c, d)]],
    ///  [],
    ///  [],
    ///  [[(g, h, i)]]]
    /// @endcode
    ///
    /// @param n The number of items in each tuple/record.
    /// @param replacement If `true`, the tuples/records are allowed to include
    /// the same item more than once, such as `(a, a, a)` and `(a, a, b)`.
    /// In the above examples, `replacement = false`. (If the output of this
    /// function is thought of as the "upper triangle" elements of the
    /// Cartesian product of the input with itself `n` times, the `replacement`
    /// parameter determines whether elements on the diagonal are allowed.)
    /// @param recordlookup If `nullptr`, the output consists of tuples, a
    /// RecordArray indexed by `"0"`, `"1"`, `"2"`, etc. If `recordlookup`
    /// is a `std::vector<std::string>`, the output consists of records,
    /// a RecordArray indexed by names (strings). The length of `recordlookup`
    /// must be equal to `n`.
    /// @param parameters Parameters assigned to the new RecordArray. This
    /// can be used to set `"__record_" = "\"record_name\\""` to give the
    /// records a custom behavior in Python.
    /// @param axis The nesting depth at which this operation is applied.
    /// At `axis = 0`, combinations are formed from elements of the whole
    /// array (see example above); at `axis = 1`, combinations are formed
    /// independently for each element. At a deeper `axis`, outer levels are
    /// left untouched.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param depth The current depth while stepping into the array: this
    /// value is set to `0` on the array node where the user starts the
    /// process and is increased at each level of list-depth (instead of
    /// decreasing the user-specified `axis`).
    ///
    /// Note that `axis = 0` is qualitatively different from any other `axis`
    /// because a dataset is typically much larger than any one of its
    /// elements. As such, `axis = 0` is lazily generated with an
    /// {@link IndexedArrayOf IndexedArray}, while any other `axis` is
    /// eagerly generated by duplicating elements (with #carry).
    virtual const ContentPtr
      combinations(int64_t n,
                   bool replacement,
                   const util::RecordLookupPtr& recordlookup,
                   const util::Parameters& parameters,
                   int64_t axis,
                   int64_t depth) const = 0;

    /// @brief Returns a string representation of this array (multi-line XML).
    ///
    /// Although this XML string has detail about every node in the tree,
    /// it does not show all elements in the array buffers and therefore
    /// does not scale with the size of the dataset (i.e. it is safe to
    /// print to the screen).
    ///
    /// Thus, it's also not a storage format: see #tojson.
    const std::string
      tostring() const;

    /// @brief Returns a JSON representation of this array.
    ///
    /// @param pretty If `true`, add spacing to make the JSON human-readable.
    /// If `false`, return a compact representation.
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for no limit.
    ///
    /// Although the JSON output contains all of the data from the array
    /// (and therefore might not be safe to print to the screen), it
    /// does not preserve the type information of an array. In particular,
    /// the distinction between ListType and RegularType is lost.
    const std::string
      tojson(bool pretty, int64_t maxdecimals) const;

    /// @brief Writes a JSON representation of this array to a `destination`
    /// file.
    ///
    /// @param destination The file to write into.
    /// @param pretty If `true`, add spacing to make the JSON human-readable.
    /// If `false`, return a compact representation.
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for no limit.
    /// @param buffersize Size of a temporary buffer in bytes.
    ///
    /// Although the JSON output contains all of the data from the array
    /// (and therefore might not be safe to print to the screen), it
    /// does not preserve the type information of an array. In particular,
    /// the distinction between ListType and RegularType is lost.
    void
      tojson(FILE* destination,
             bool pretty,
             int64_t maxdecimals,
             int64_t buffersize) const;

    /// @brief The number of bytes contained in all array buffers,
    /// {@link IndexOf Index} buffers, and Identities buffers, not including
    /// the lightweight node objects themselves.
    int64_t
      nbytes() const;

    /// @brief This array with one axis removed by applying a Reducer
    /// (e.g. "sum", "max", "any", "all).
    ///
    /// This operation is implemented on each node through #reduce_next.
    ///
    /// @param reducer The choice of Reducer algorithm.
    /// @param axis The axis to remove by reduction.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param mask If `true`, the Reducer's identity values will be covered
    /// by `None` using a ByteMaskedArray. This is desirable for ReducerMin,
    /// ReducerMax, ReducerArgmin, and ReducerArgmax to indicate that empty
    /// lists have no minimum or maximum.
    /// @param keepdims If `true`, the reduced values will be wrapped by a
    /// singleton RegularArray to maintain the same number of dimensions in
    /// the output.
    const ContentPtr
      reduce(const Reducer& reducer,
             int64_t axis,
             bool mask,
             bool keepdims) const;

    /// @brief This array with one axis sorted by applying a sorting algorithm
    ///
    /// This operation is implemented on each node through #sort_next.
    ///
    /// @param axis The axis to sort.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param ascending If `true`, the values will be sorted in an ascending
    /// order.
    /// @param stable If `true`, the values will be sorted by a
    /// stable sort algorithm to maintain the relative order of records with
    /// equal keys (i.e. values).
    const ContentPtr
      sort(int64_t axis, bool ascending, bool stable) const;

    /// @brief This array indices with one axis sorted by applying
    ///        a sorting algorithm
    ///
    /// This operation is implemented on each node through #argsort_next.
    ///
    /// @param axis The axis to sort.
    /// Negative `axis` counts backward from the deepest levels (`-1` is
    /// the last valid `axis`).
    /// @param ascending If `true`, the values will be sorted in an ascending
    /// order.
    /// @param stable If `true`, the values will be sorted by a
    /// stable sort algorithm to maintain the relative order of records with
    /// equal keys (i.e. values).
    const ContentPtr
      argsort(int64_t axis, bool ascending, bool stable) const;

    /// @brief String-to-JSON map that augments the meaning of this
    /// array.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const util::Parameters
      parameters() const;

    /// @brief Assign all the parameters for this array node (in-place).
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// @note This mutability is temporary:
    /// [scikit-hep/awkward-1.0#117](https://github.com/scikit-hep/awkward-1.0/issues/177)
    /// Eventually, this interface will be deprecated and all Content
    /// instances will be immutable.
    void
      setparameters(const util::Parameters& parameters);

    /// @brief Get one parameter from this array node.
    ///
    /// If the `key` does not exist, this function returns `"null"`.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const std::string
      parameter(const std::string& key) const;

    /// @brief Assign one parameter for this array node (in-place).
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// @note This mutability is temporary:
    /// [scikit-hep/awkward-1.0#117](https://github.com/scikit-hep/awkward-1.0/issues/177)
    /// Eventually, this interface will be deprecated and all Content
    /// instances will be immutable.
    void
      setparameter(const std::string& key, const std::string& value);

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

    /// @brief Returns `true` if all parameters of this array node are equal
    /// to the `other` parameters.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    ///
    /// Equality is checked at the level of JSON DOMs. The `value` does not
    /// need to be exactly the same string; it needs to have equivalent JSON
    /// value.
    bool
      parameters_equal(const util::Parameters& other) const;

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string; `false` otherwise.
    bool
      parameter_isstring(const std::string& key) const;

    /// @brief Returns `true` if the parameter associated with `key` is a
    /// string that matches `[A-Za-z_][A-Za-z_0-9]*`; `false` otherwise.
    bool
      parameter_isname(const std::string& key) const;

    /// @brief Returns the parameter associated with `key` as a string if
    /// #parameter_isstring; raises an error otherwise.
    const std::string
      parameter_asstring(const std::string& key) const;

    /// @brief Concatenates this array with `other` by creating a
    /// {@link UnionArrayOf UnionArray} instead of actually merging the data.
    const ContentPtr
      merge_as_union(const ContentPtr& other) const;

    /// @brief Internal function to handle the `axis = 0` case of #rpad
    /// and #rpad_and_clip.
    ///
    /// The `axis = 0` case does not depend on array node type, so it is
    /// defined universally in the Content class.
    ///
    /// @param target The intended length. The output may be longer than this
    /// target length, but not shorter (using {@link ListArrayOf ListArray}).
    /// @param clip If `true`, do #rpad_and_clip; if `false`, do #rpad.
    const ContentPtr
      rpad_axis0(int64_t target, bool clip) const;

    /// @brief Internal function to handle the `axis = 0` case of #localindex.
    ///
    /// The `axis = 0` case does not depend on array node type, so it is
    /// defined universally in the Content class.
    const ContentPtr
      localindex_axis0() const;

    /// @brief Internal function to handle the `axis = 0` case of
    /// #combinations.
    ///
    /// The `axis = 0` case does not depend on array node type, so it is
    /// defined universally in the Content class.
    ///
    /// @param n The number of items in each tuple/record.
    /// @param replacement If `true`, the tuples/records are allowed to include
    /// the same item more than once, such as `(a, a, a)` and `(a, a, b)`.
    /// In the above examples, `replacement = false`. (If the output of this
    /// function is thought of as the "upper triangle" elements of the
    /// Cartesian product of the input with itself `n` times, the `replacement`
    /// parameter determines whether elements on the diagonal are allowed.)
    /// @param recordlookup If `nullptr`, the output consists of tuples, a
    /// RecordArray indexed by `"0"`, `"1"`, `"2"`, etc. If `recordlookup`
    /// is a `std::vector<std::string>`, the output consists of records,
    /// a RecordArray indexed by names (strings). The length of `recordlookup`
    /// must be equal to `n`.
    /// @param parameters Parameters assigned to the new RecordArray. This
    /// can be used to set `"__record_" = "\"record_name\\""` to give the
    /// records a custom behavior in Python.
    const ContentPtr
      combinations_axis0(int64_t n,
                         bool replacement,
                         const util::RecordLookupPtr& recordlookup,
                         const util::Parameters& parameters) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceEllipsis& ellipsis,
                   const Slice& tail,
                   const Index64& advanced) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceNewAxis& newaxis,
                   const Slice& tail,
                   const Index64& advanced) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceField& field,
                   const Slice& tail,
                   const Index64& advanced) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceFields& fields,
                   const Slice& tail,
                   const Index64& advanced) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceMissing64& missing,
                   const Slice& tail,
                   const Index64& advanced) const;

    /// @brief Internal function that propagates a generic #getitem request
    /// through one axis (including advanced indexing).
    ///
    /// See generic #getitem_next for details.
    virtual const ContentPtr
      getitem_next(const SliceJagged64& jagged,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    /// @brief Internal function that propagates a jagged array (array with
    /// irregular-length dimensions) slice through one axis.
    ///
    /// See generic #getitem_next_jagged for details.
    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceArray64& slicecontent,
                          const Slice& tail) const = 0;

    /// @brief Internal function that propagates a jagged array (array with
    /// irregular-length dimensions) slice through one axis.
    ///
    /// See generic #getitem_next_jagged for details.
    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceMissing64& slicecontent,
                          const Slice& tail) const = 0;

    /// @brief Internal function that propagates a jagged array (array with
    /// irregular-length dimensions) slice through one axis.
    ///
    /// See generic #getitem_next_jagged for details.
    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceJagged64& slicecontent,
                          const Slice& tail) const = 0;

    /// @brief Internal function defining the negative axis handling for many
    /// operations.
    ///
    /// Returns a non-negative equivalent `axis` if unambiguous and passes
    /// the negative `axis` through if ambiguous. A negative `axis` can be
    /// ambiguous if the list-depth "branches" in a RecordArray or a
    /// {@link UnionArrayOf UnionArray} with different `contents` having
    /// different depths. As an operation descends through the nodes of
    /// an array, it repeatedly calls `axis_wrap_if_negative` until an
    /// unambiguous non-negative `axis` can be identified.
    ///
    /// This allows a RecordArray or {@link UnionArrayOf UnionArray} with
    /// different depths to accept `axis = -1` as the last axis, regardless
    /// of how deep that is in different record fields or union possibilities.
    const int64_t
      axis_wrap_if_negative(int64_t axis) const;

    /// @brief Recursively copies components of the array from main memory to a
    /// GPU (if `ptr_lib == kernel::lib::cuda`) or to main memory (if
    /// `ptr_lib == kernel::lib::cpu`) if those components are not already there.
    virtual const ContentPtr
      copy_to(kernel::lib ptr_lib) const = 0;

    /// @brief Change the leaf types to 'totype'.
    virtual const ContentPtr
      numbers_to_type(const std::string& name) const = 0;

  protected:
    /// @brief Internal function to wrap putative #getitem output with enough
    /// RegularArray nodes to satisfy a given `shape`.
    ///
    /// The `shape` is intended to match a
    /// {@link SliceArrayOf#shape SliceArray::shape}.
    const ContentPtr
      getitem_next_array_wrap(const ContentPtr& outcontent,
                              const std::vector<int64_t>& shape) const;

    /// @brief Internal function to convert #parameters into a string fragment
    /// for #tostring.
    ///
    /// @param indent Indentation depth as a string of spaces.
    /// @param pre Prefix string, usually an opening XML tag.
    /// @param post Postfix string, usually a closing XML tag and carriage
    /// return.
    const std::string
      parameters_tostring(const std::string& indent,
                          const std::string& pre,
                          const std::string& post) const;

  protected:
    /// @brief See #identities.
    IdentitiesPtr identities_;
    /// @brief See #parameters.
    util::Parameters parameters_;
  };
}

#endif // AWKWARD_CONTENT_H_

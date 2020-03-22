// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CONTENT_H_
#define AWKWARD_CONTENT_H_

#include <cstdio>
#include <map>

#include "awkward/cpu-kernels/util.h"
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

  class EXPORT_SYMBOL Content {
  public:
    Content(const IdentitiesPtr& identities,
            const util::Parameters& parameters);

    virtual ~Content();

    virtual bool
      isscalar() const;

    virtual const std::string
      classname() const = 0;

    virtual const IdentitiesPtr
      identities() const;

    virtual void
      setidentities() = 0;

    virtual void
      setidentities(const IdentitiesPtr& identities) = 0;

    virtual const TypePtr
      type(const util::TypeStrs& typestrs) const = 0;

    virtual const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;

    virtual void
      tojson_part(ToJson& builder) const = 0;

    virtual void
      nbytes_part(std::map<size_t, int64_t>& largest) const = 0;

    virtual int64_t
      length() const = 0;

    virtual const ContentPtr
      shallow_copy() const = 0;

    virtual const ContentPtr
      deep_copy(bool copyarrays,
                bool copyindexes,
                bool copyidentities) const = 0;

    /// Performs up-front validity checks on an array so that they don't have
    /// to be checked in #getitem_at_nowrap for each item.
    virtual void
      check_for_iteration() const = 0;

    virtual const ContentPtr
      getitem_nothing() const = 0;

    virtual const ContentPtr
      getitem_at(int64_t at) const = 0;

    virtual const ContentPtr
      getitem_at_nowrap(int64_t at) const = 0;

    virtual const ContentPtr
      getitem_range(int64_t start, int64_t stop) const = 0;

    virtual const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const = 0;

    virtual const ContentPtr
      getitem_field(const std::string& key) const = 0;

    virtual const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const = 0;

    virtual const ContentPtr
      getitem(const Slice& where) const;

    virtual const ContentPtr
      getitem_next(const SliceItemPtr& head,
                   const Slice& tail,
                   const Index64& advanced) const;

    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceItemPtr& slicecontent,
                          const Slice& tail) const;

    /// Returns an array of the same type with elements selected, rearranged,
    /// and possibly duplicated by the `carry` array of integers.
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
    /// selections/rearrangements/duplications from one typed array node to
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
      carry(const Index64& carry) const = 0;

    virtual const std::string
      purelist_parameter(const std::string& key) const = 0;

    virtual bool
      purelist_isregular() const = 0;

    virtual int64_t
      purelist_depth() const = 0;

    virtual const std::pair<int64_t, int64_t>
      minmax_depth() const = 0;

    /// Returns (a) whether the list-depth of this array "branches," or differs
    /// when followed through different fields of a RecordArray or UnionArray,
    /// and (b) the minimum list-depth.
    ///
    /// If the array does not contain any records or heterogeneous data, the
    /// `first` element is always `true` and the `second` is simply the depth.
    virtual const std::pair<bool, int64_t>
      branch_depth() const = 0;

    virtual int64_t
      numfields() const = 0;

    virtual int64_t
      fieldindex(const std::string& key) const = 0;

    virtual const std::string
      key(int64_t fieldindex) const = 0;

    virtual bool
      haskey(const std::string& key) const = 0;

    virtual const std::vector<std::string>
      keys() const = 0;

    // operations
    virtual const std::string
      validityerror(const std::string& path) const = 0;

    virtual const ContentPtr
      shallow_simplify() const = 0;

    virtual const ContentPtr
      num(int64_t axis, int64_t depth) const = 0;

    virtual const std::pair<Index64, ContentPtr>
      offsets_and_flattened(int64_t axis, int64_t depth) const = 0;

    virtual bool
      mergeable(const ContentPtr& other, bool mergebool) const = 0;

    virtual const ContentPtr
      merge(const ContentPtr& other) const = 0;

    /// Converts this array into a SliceItem that can be used in getitem.
    virtual const SliceItemPtr
      asslice() const = 0;

    virtual const ContentPtr
      fillna(const ContentPtr& value) const = 0;

    virtual const ContentPtr
      rpad(int64_t length, int64_t axis, int64_t depth) const = 0;

    virtual const ContentPtr
      rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const = 0;

    virtual const ContentPtr
      reduce_next(const Reducer& reducer,
                  int64_t negaxis,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength,
                  bool mask,
                  bool keepdims) const = 0;

    virtual const ContentPtr
      localindex(int64_t axis, int64_t depth) const = 0;

    /// Returns tuples or records of all `n`-tuple combinations of list items
    /// at some `axis` depth.
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
    /// @param diagonal If `true`, the tuples/records are allowed to include
    /// the same item more than once, such as `(a, a, a)` and `(a, a, b)`.
    /// In the above examples, `diagonal = false`. (If the output of this
    /// function is thought of as the "upper triangle" elements of the
    /// Cartesian product of the input with itself `n` times, the `diagonal`
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
    /// @note `axis = 0` is qualitatively different from any other `axis`
    /// because a dataset is typically much larger than any one of its
    /// elements. As such, `axis = 0` is lazily generated with an
    /// {@link IndexedArrayOf IndexedArray}, while any other `axis` is
    /// eagerly generated by duplicating elements (with #carry).
    virtual const ContentPtr
      choose(int64_t n,
             bool diagonal,
             const util::RecordLookupPtr& recordlookup,
             const util::Parameters& parameters,
             int64_t axis,
             int64_t depth) const = 0;

    const std::string
      tostring() const;

    const std::string
      tojson(bool pretty, int64_t maxdecimals) const;

    void
      tojson(FILE* destination,
             bool pretty,
             int64_t maxdecimals,
             int64_t buffersize) const;

    int64_t
      nbytes() const;

    const ContentPtr
      reduce(const Reducer& reducer,
             int64_t axis,
             bool mask,
             bool keepdims) const;

    const util::Parameters
      parameters() const;

    void
      setparameters(const util::Parameters& parameters);

    const std::string
      parameter(const std::string& key) const;

    void
      setparameter(const std::string& key, const std::string& value);

    bool
      parameter_equals(const std::string& key, const std::string& value) const;

    bool
      parameters_equal(const util::Parameters& other) const;

    const ContentPtr
      merge_as_union(const ContentPtr& other) const;

    const ContentPtr
      rpad_axis0(int64_t target, bool clip) const;

    const ContentPtr
      localindex_axis0() const;

    const ContentPtr
      choose_axis0(int64_t n,
                   bool diagonal,
                   const util::RecordLookupPtr& recordlookup,
                   const util::Parameters& parameters) const;

    virtual const ContentPtr
      getitem_next(const SliceAt& at,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    virtual const ContentPtr
      getitem_next(const SliceRange& range,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    virtual const ContentPtr
      getitem_next(const SliceEllipsis& ellipsis,
                   const Slice& tail,
                   const Index64& advanced) const;

    virtual const ContentPtr
      getitem_next(const SliceNewAxis& newaxis,
                   const Slice& tail,
                   const Index64& advanced) const;

    virtual const ContentPtr
      getitem_next(const SliceArray64& array,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    virtual const ContentPtr
      getitem_next(const SliceField& field,
                   const Slice& tail,
                   const Index64& advanced) const;

    virtual const ContentPtr
      getitem_next(const SliceFields& fields,
                   const Slice& tail,
                   const Index64& advanced) const;

    virtual const ContentPtr
      getitem_next(const SliceMissing64& missing,
                   const Slice& tail,
                   const Index64& advanced) const;

    virtual const ContentPtr
      getitem_next(const SliceJagged64& jagged,
                   const Slice& tail,
                   const Index64& advanced) const = 0;

    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceArray64& slicecontent,
                          const Slice& tail) const = 0;

    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceMissing64& slicecontent,
                          const Slice& tail) const = 0;

    virtual const ContentPtr
      getitem_next_jagged(const Index64& slicestarts,
                          const Index64& slicestops,
                          const SliceJagged64& slicecontent,
                          const Slice& tail) const = 0;

  protected:
    const ContentPtr
      getitem_next_array_wrap(const ContentPtr& outcontent,
                              const std::vector<int64_t>& shape) const;

    const std::string
      parameters_tostring(const std::string& indent,
                          const std::string& pre,
                          const std::string& post) const;

    const int64_t
      axis_wrap_if_negative(int64_t axis) const;

  protected:
    IdentitiesPtr identities_;
    util::Parameters parameters_;
  };
}

#endif // AWKWARD_CONTENT_H_

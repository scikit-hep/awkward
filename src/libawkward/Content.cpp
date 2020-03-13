// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/reducers.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/ArrayType.h"

#include "awkward/Content.h"

namespace awkward {
  Content::Content(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters)
      : identities_(identities)
      , parameters_(parameters) { }

  Content::~Content() { }

  bool Content::isscalar() const {
    return false;
  }

  const std::shared_ptr<Identities> Content::identities() const {
    return identities_;
  }

  const std::string Content::tostring() const {
    return tostring_part("", "", "");
  }

  const std::string Content::tojson(bool pretty, int64_t maxdecimals) const {
    if (pretty) {
      ToJsonPrettyString builder(maxdecimals);
      tojson_part(builder);
      return builder.tostring();
    }
    else {
      ToJsonString builder(maxdecimals);
      tojson_part(builder);
      return builder.tostring();
    }
  }

  void Content::tojson(FILE* destination, bool pretty, int64_t maxdecimals, int64_t buffersize) const {
    if (pretty) {
      ToJsonPrettyFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder);
      builder.endlist();
    }
    else {
      ToJsonFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      tojson_part(builder);
      builder.endlist();
    }
  }

  int64_t Content::nbytes() const {
    // FIXME: this is only accurate if all subintervals of allocated arrays are nested
    // (which is likely, but not guaranteed). In general, it's <= the correct nbytes.
    std::map<size_t, int64_t> largest;
    nbytes_part(largest);
    int64_t out = 0;
    for (auto pair : largest) {
      out += pair.second;
    }
    return out;
  }

  const std::shared_ptr<Content> Content::reduce(const Reducer& reducer, int64_t axis, bool mask, bool keepdims) const {
    int64_t negaxis = -axis;
    std::pair<bool, int64_t> branchdepth = branch_depth();
    bool branch = branchdepth.first;
    int64_t depth = branchdepth.second;

    if (branch) {
      if (negaxis <= 0) {
        throw std::invalid_argument("cannot use non-negative axis on a nested list structure of variable depth (negative axis counts from the leaves of the tree; non-negative from the root)");
      }
      if (negaxis > depth) {
        throw std::invalid_argument(std::string("cannot use axis=") + std::to_string(axis) + std::string(" on a nested list structure that splits into different depths, the minimum of which is depth=") + std::to_string(depth) + std::string(" from the leaves"));
      }
    }
    else {
      if (negaxis <= 0) {
        negaxis += depth;
      }
      if (!(0 < negaxis  &&  negaxis <= depth)) {
        throw std::invalid_argument(std::string("axis=") + std::to_string(axis) + std::string(" exceeds the depth of the nested list structure (which is ") + std::to_string(depth) + std::string(")"));
      }
    }

    Index64 parents(length());
    struct Error err = awkward_content_reduce_zeroparents_64(
      parents.ptr().get(),
      length());
    util::handle_error(err, classname(), identities_.get());

    std::shared_ptr<Content> next = reduce_next(reducer, negaxis, parents, 1, mask, keepdims);
    return next.get()->getitem_at_nowrap(0);
  }

  const util::Parameters Content::parameters() const {
    return parameters_;
  }

  void Content::setparameters(const util::Parameters& parameters) {
    parameters_ = parameters;
  }

  const std::string Content::parameter(const std::string& key) const {
    auto item = parameters_.find(key);
    if (item == parameters_.end()) {
      return "null";
    }
    return item->second;
  }

  void Content::setparameter(const std::string& key, const std::string& value) {
    parameters_[key] = value;
  }

  bool Content::parameter_equals(const std::string& key, const std::string& value) const {
    return util::parameter_equals(parameters_, key, value);
  }

  bool Content::parameters_equal(const util::Parameters& other) const {
    return util::parameters_equal(parameters_, other);
  }

  const std::shared_ptr<Content> Content::merge_as_union(const std::shared_ptr<Content>& other) const {
    int64_t mylength = length();
    int64_t theirlength = other.get()->length();
    Index8 tags(mylength + theirlength);
    Index64 index(mylength + theirlength);

    std::vector<std::shared_ptr<Content>> contents({ shallow_copy(), other });

    struct Error err1 = awkward_unionarray_filltags_to8_const(
      tags.ptr().get(),
      0,
      mylength,
      0);
    util::handle_error(err1, classname(), identities_.get());
    struct Error err2 = awkward_unionarray_fillindex_to64_count(
      index.ptr().get(),
      0,
      mylength);
    util::handle_error(err2, classname(), identities_.get());

    struct Error err3 = awkward_unionarray_filltags_to8_const(
      tags.ptr().get(),
      mylength,
      theirlength,
      1);
    util::handle_error(err3, classname(), identities_.get());
    struct Error err4 = awkward_unionarray_fillindex_to64_count(
      index.ptr().get(),
      mylength,
      theirlength);
    util::handle_error(err4, classname(), identities_.get());

    return std::make_shared<UnionArray8_64>(Identities::none(), util::Parameters(), tags, index, contents);
  }

  const std::shared_ptr<Content> Content::rpad_axis0(int64_t target, bool clip) const {
    if (!clip  &&  target < length()) {
      return shallow_copy();
    }
    Index64 index(target);
    struct Error err = awkward_index_rpad_and_clip_axis0_64(
      index.ptr().get(),
      target,
      length());
    util::handle_error(err, classname(), identities_.get());
    std::shared_ptr<IndexedOptionArray64> next = std::make_shared<IndexedOptionArray64>(Identities::none(), util::Parameters(), index, shallow_copy());
    return next.get()->simplify();
  }

  const std::shared_ptr<Content> Content::localindex_axis0() const {
    Index64 localindex(length());
    struct Error err = awkward_localindex_64(
      localindex.ptr().get(),
      length());
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<NumpyArray>(localindex);
  }

  const std::shared_ptr<Content> Content::choose_axis0(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters) const {
    int64_t size = length();
    if (diagonal) {
      size += (n - 1);
    }
    int64_t thisn = n;
    int64_t chooselen;
    if (thisn > size) {
      chooselen = 0;
    }
    else if (thisn == size) {
      chooselen = 1;
    }
    else {
      if (thisn * 2 > size) {
        thisn = size - thisn;
      }
      chooselen = size;
      for (int64_t j = 2;  j <= thisn;  j++) {
        chooselen *= (size - j + 1);
        chooselen /= j;
      }
    }

    std::vector<std::shared_ptr<int64_t>> tocarry;
    std::vector<int64_t*> tocarryraw;
    for (int64_t j = 0;  j < n;  j++) {
      std::shared_ptr<int64_t> ptr(new int64_t[(size_t)chooselen], util::array_deleter<int64_t>());
      tocarry.push_back(ptr);
      tocarryraw.push_back(ptr.get());
    }
    struct Error err = awkward_regulararray_choose_64(
      tocarryraw.data(),
      n,
      diagonal,
      length(),
      1);
    util::handle_error(err, classname(), identities_.get());

    std::vector<std::shared_ptr<Content>> contents;
    for (auto ptr : tocarry) {
      contents.push_back(std::make_shared<IndexedArray64>(Identities::none(), util::Parameters(), Index64(ptr, 0, chooselen), shallow_copy()));
    }
    return std::make_shared<RecordArray>(Identities::none(), parameters, contents, recordlookup);
  }

  const std::shared_ptr<Content> Content::getitem(const Slice& where) const {
    std::shared_ptr<Content> next = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), shallow_copy(), length());

    std::shared_ptr<SliceItem> nexthead = where.head();
    Slice nexttail = where.tail();
    Index64 nextadvanced(0);
    std::shared_ptr<Content> out = next.get()->getitem_next(nexthead, nexttail, nextadvanced);

    if (out.get()->length() == 0) {
      return out.get()->getitem_nothing();
    }
    else {
      return out.get()->getitem_at_nowrap(0);
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      return getitem_next(*at, tail, advanced);
    }
    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      return getitem_next(*range, tail, advanced);
    }
    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      return getitem_next(*array, tail, advanced);
    }
    else if (SliceField* field = dynamic_cast<SliceField*>(head.get())) {
      return getitem_next(*field, tail, advanced);
    }
    else if (SliceFields* fields = dynamic_cast<SliceFields*>(head.get())) {
      return getitem_next(*fields, tail, advanced);
    }
    else if (SliceMissing64* missing = dynamic_cast<SliceMissing64*>(head.get())) {
      return getitem_next(*missing, tail, advanced);
    }
    else if (SliceJagged64* jagged = dynamic_cast<SliceJagged64*>(head.get())) {
      return getitem_next(*jagged, tail, advanced);
    }
    else {
      throw std::runtime_error("unrecognized slice type");
    }
  }

  const std::shared_ptr<Content> Content::getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const std::shared_ptr<SliceItem>& slicecontent, const Slice& tail) const {
    if (SliceArray64* array = dynamic_cast<SliceArray64*>(slicecontent.get())) {
      return getitem_next_jagged(slicestarts, slicestops, *array, tail);
    }
    else if (SliceMissing64* missing = dynamic_cast<SliceMissing64*>(slicecontent.get())) {
      return getitem_next_jagged(slicestarts, slicestops, *missing, tail);
    }
    else if (SliceJagged64* jagged = dynamic_cast<SliceJagged64*>(slicecontent.get())) {
      return getitem_next_jagged(slicestarts, slicestops, *jagged, tail);
    }
    else {
      throw std::runtime_error("unexpected slice type for getitem_next_jagged");
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceEllipsis& ellipsis, const Slice& tail, const Index64& advanced) const {
    std::pair<int64_t, int64_t> minmax = minmax_depth();
    int64_t mindepth = minmax.first;
    int64_t maxdepth = minmax.second;

    if (tail.length() == 0  ||  (mindepth - 1 == tail.dimlength()  &&  maxdepth - 1 == tail.dimlength())) {
      std::shared_ptr<SliceItem> nexthead = tail.head();
      Slice nexttail = tail.tail();
      return getitem_next(nexthead, nexttail, advanced);
    }
    else if (mindepth - 1 == tail.dimlength()  ||  maxdepth - 1 == tail.dimlength()) {
      throw std::invalid_argument("ellipsis (...) can't be used on a data structure of different depths");
    }
    else {
      std::vector<std::shared_ptr<SliceItem>> tailitems = tail.items();
      std::vector<std::shared_ptr<SliceItem>> items = { std::make_shared<SliceEllipsis>() };
      items.insert(items.end(), tailitems.begin(), tailitems.end());
      std::shared_ptr<SliceItem> nexthead = std::make_shared<SliceRange>(Slice::none(), Slice::none(), 1);
      Slice nexttail(items);
      return getitem_next(nexthead, nexttail, advanced);
    }
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceNewAxis& newaxis, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return std::make_shared<RegularArray>(Identities::none(), util::Parameters(), getitem_next(nexthead, nexttail, advanced), 1);
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_field(field.key()).get()->getitem_next(nexthead, nexttail, advanced);
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_fields(fields.keys()).get()->getitem_next(nexthead, nexttail, advanced);
  }

  const std::shared_ptr<Content> getitem_next_regular_missing(const SliceMissing64& missing, const Slice& tail, const Index64& advanced, const RegularArray* raw, int64_t length, const std::string& classname) {
    Index64 index(missing.index());
    Index64 outindex(index.length()*length);

    struct Error err = awkward_missing_repeat_64(
      outindex.ptr().get(),
      index.ptr().get(),
      index.offset(),
      index.length(),
      length,
      raw->size());
    util::handle_error(err, classname, nullptr);

    IndexedOptionArray64 out(Identities::none(), util::Parameters(), outindex, raw->content());
    return std::make_shared<RegularArray>(Identities::none(), util::Parameters(), out.simplify(), index.length());
  }

  const std::shared_ptr<Content> Content::getitem_next(const SliceMissing64& missing, const Slice& tail, const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::invalid_argument("cannot mix missing values in slice with NumPy-style advanced indexing");
    }

    std::shared_ptr<Content> next = getitem_next(missing.content(), tail, advanced);

    if (RegularArray* raw = dynamic_cast<RegularArray*>(next.get())) {
      return getitem_next_regular_missing(missing, tail, advanced, raw, length(), classname());
    }

    else if (RecordArray* rec = dynamic_cast<RecordArray*>(next.get())) {
      if (rec->numfields() == 0) {
        return next;
      }
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : rec->contents()) {
        if (RegularArray* raw = dynamic_cast<RegularArray*>(content.get())) {
          contents.push_back(getitem_next_regular_missing(missing, tail, advanced, raw, length(), classname()));
        }
        else {
          throw std::runtime_error(std::string("FIXME: unhandled case of SliceMissing with RecordArray containing\n") + content.get()->tostring());
        }
      }
      return std::make_shared<RecordArray>(Identities::none(), util::Parameters(), contents, rec->recordlookup());
    }

    else {
      throw std::runtime_error(std::string("FIXME: unhandled case of SliceMissing with\n") + next.get()->tostring());
    }
  }

  const std::shared_ptr<Content> Content::getitem_next_array_wrap(const std::shared_ptr<Content>& outcontent, const std::vector<int64_t>& shape) const {
    std::shared_ptr<Content> out = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), outcontent, (int64_t)shape[shape.size() - 1]);
    for (int64_t i = (int64_t)shape.size() - 2;  i >= 0;  i--) {
      out = std::make_shared<RegularArray>(Identities::none(), util::Parameters(), out, (int64_t)shape[(size_t)i]);
    }
    return out;
  }

  const std::string Content::parameters_tostring(const std::string& indent, const std::string& pre, const std::string& post) const {
    if (parameters_.empty()) {
      return "";
    }
    else {
      std::stringstream out;
      out << indent << pre << "<parameters>\n";
      for (auto pair : parameters_) {
        out << indent << "    <param key=" << util::quote(pair.first, true) << ">" << pair.second << "</param>\n";
      }
      out << indent << "</parameters>" << post;
      return out.str();
    }
  }

  const int64_t Content::axis_wrap_if_negative(int64_t axis) const {
    if (axis < 0) {
      throw std::runtime_error("FIXME: negative axis not implemented yet");
    }
    return axis;
  }
}

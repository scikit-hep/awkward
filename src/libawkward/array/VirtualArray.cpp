// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/array/VirtualArray.h"

namespace awkward {
  ////////// VirtualArray

  VirtualArray::VirtualArray(const IdentitiesPtr& identities,
                             const util::Parameters& parameters,
                             const ArrayGeneratorPtr& generator,
                             const ArrayCachePtr& cache,
                             const std::string& cache_key)
      : Content(identities, parameters)
      , generator_(generator)
      , cache_(cache)
      , cache_key_(cache_key) { }

  VirtualArray::VirtualArray(const IdentitiesPtr& identities,
                             const util::Parameters& parameters,
                             const ArrayGeneratorPtr& generator,
                             const ArrayCachePtr& cache)
      : Content(identities, parameters)
      , generator_(generator)
      , cache_(cache)
      , cache_key_(ArrayCache::newkey()) { }

  const ArrayGeneratorPtr
  VirtualArray::generator() const {
    return generator_;
  }

  const ArrayCachePtr
  VirtualArray::cache() const {
    return cache_;
  }

  const ContentPtr
  VirtualArray::peek_array() const {
    if (cache_.get() != nullptr) {
      return cache_.get()->get(cache_key());
    }
    return ContentPtr(nullptr);
  }

  const ContentPtr
  VirtualArray::array() const {
    ContentPtr out(nullptr);
    if (cache_.get() != nullptr) {
      out = cache_.get()->get(cache_key());
    }
    if (out.get() == nullptr) {
      out = generator_.get()->generate_and_check();
    }
    if (cache_.get() != nullptr) {
      cache_.get()->set(cache_key(), out);
    }
    return out;
  }

  const std::string
  VirtualArray::cache_key() const {
    return cache_key_;
  }

  const std::string
  VirtualArray::classname() const {
    return "VirtualArray";
  }

  void
  VirtualArray::setidentities(const IdentitiesPtr& identities) {
    throw std::runtime_error("FIXME: VirtualArray::setidentities(identities)");
  }

  void
  VirtualArray::setidentities() {
    throw std::runtime_error("FIXME: VirtualArray::setidentities");
  }

  const TypePtr
  VirtualArray::type(const util::TypeStrs& typestrs) const {
    return form().get()->type(typestrs);
  }

  const FormPtr
  VirtualArray::form() const {
    FormPtr out = generator_.get()->form();
    if (out.get() == nullptr) {
      out = array().get()->form();
    }
    return out;
  }

  const std::string
  VirtualArray::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname()
        << " cache_key=\"" << cache_key_ << "\">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(
               indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << generator_.get()->tostring_part(indent + std::string("    "),
                                           "", "\n");
    if (cache_.get() != nullptr) {
      out << cache_.get()->tostring_part(indent + std::string("    "),
                                         "", "\n");
    }
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      out << peek.get()->tostring_part(
               indent + std::string("    "), "<array>", "</array>\n");
    }
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void
  VirtualArray::tojson_part(ToJson& builder,
                          bool include_beginendlist) const {
    return array().get()->tojson_part(builder, include_beginendlist);
  }

  void
  VirtualArray::nbytes_part(std::map<size_t, int64_t>& largest) const { }

  int64_t
  VirtualArray::length() const {
    int64_t out = generator_.get()->length();
    if (out < 0) {
      out = array().get()->length();
    }
    return out;
  }

  const ContentPtr
  VirtualArray::shallow_copy() const {
    return std::make_shared<VirtualArray>(identities_,
                                          parameters_,
                                          generator_,
                                          cache_,
                                          cache_key_);
  }

  const ContentPtr
  VirtualArray::deep_copy(bool copyarrays,
                          bool copyindexes,
                          bool copyidentities) const {
    return array().get()->deep_copy(copyarrays, copyindexes, copyidentities);
  }

  void
  VirtualArray::check_for_iteration() const { }

  const ContentPtr
  VirtualArray::getitem_nothing() const {
    return array().get()->getitem_nothing();
  }

  const ContentPtr
  VirtualArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length();
    }
    if (!(0 <= regular_at  &&  regular_at < length())) {
      util::handle_error(failure("index out of range", kSliceNone, at),
                         classname(),
                         identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  VirtualArray::getitem_at_nowrap(int64_t at) const {
    return array().get()->getitem_at_nowrap(at);
  }

  const ContentPtr
  VirtualArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(),
      length());
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(failure("index out of range", kSliceNone, stop),
                         identities_.get()->classname(),
                         nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const ContentPtr
  VirtualArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    Slice slice;
    slice.append(SliceRange(start, stop, 1));
    slice.become_sealed();
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 generator_.get()->form(), stop - start, generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          parameters_,
                                          generator,
                                          cache);
  }

  const ContentPtr
  VirtualArray::getitem_field(const std::string& key) const {
    Slice slice;
    slice.append(SliceField(key));
    slice.become_sealed();
    FormPtr form = generator_.get()->form();
    if (form.get() != nullptr) {
      form = form.get()->getitem_field(key);
    }
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 form, length(), generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          parameters_,
                                          generator,
                                          cache);
  }

  const ContentPtr
  VirtualArray::getitem_fields(const std::vector<std::string>& keys) const {
    Slice slice;
    slice.append(SliceFields(keys));
    slice.become_sealed();
    FormPtr form = generator_.get()->form();
    if (form.get() != nullptr) {
      form = form.get()->getitem_fields(keys);
    }
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 form, length(), generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          parameters_,
                                          generator,
                                          cache);
  }

  const ContentPtr
  VirtualArray::carry(const Index64& carry) const {
    Slice slice;
    std::vector<int64_t> shape({ carry.length() });
    std::vector<int64_t> strides({ 1 });
    slice.append(SliceArray64(carry, shape, strides, false));
    slice.become_sealed();
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 generator_.get()->form(), carry.length(), generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          parameters_,
                                          generator,
                                          cache);
  }

  const std::string
  VirtualArray::validityerror(const std::string& path) const {
    throw std::runtime_error("FIXME: VirtualArray::validityerror");
  }

  const ContentPtr
  VirtualArray::shallow_simplify() const {
    throw std::runtime_error("FIXME: VirtualArray::shallow_simplify");
  }

  const ContentPtr
  VirtualArray::num(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: VirtualArray::num");
  }

  const std::pair<Index64, ContentPtr>
  VirtualArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: VirtualArray::offsets_and_flattened");
  }

  bool
  VirtualArray::mergeable(const ContentPtr& other, bool mergebool) const {
    throw std::runtime_error("FIXME: VirtualArray::mergeable");
  }

  const ContentPtr
  VirtualArray::merge(const ContentPtr& other) const {
    throw std::runtime_error("FIXME: VirtualArray::merge");
  }

  const SliceItemPtr
  VirtualArray::asslice() const {
    throw std::runtime_error("FIXME: VirtualArray::asslice");
  }

  const ContentPtr
  VirtualArray::fillna(const ContentPtr& value) const {
    throw std::runtime_error("FIXME: VirtualArray::fillna");
  }

  const ContentPtr
  VirtualArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: VirtualArray::rpad");
  }

  const ContentPtr
  VirtualArray::rpad_and_clip(int64_t target,
                            int64_t axis,
                            int64_t depth) const {
    throw std::runtime_error("FIXME: VirtualArray::rpad_and_clip");
  }

  const ContentPtr
  VirtualArray::reduce_next(const Reducer& reducer,
                          int64_t negaxis,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength,
                          bool mask,
                          bool keepdims) const {
    throw std::runtime_error("FIXME: VirtualArray::reduce_next");
  }

  const ContentPtr
  VirtualArray::localindex(int64_t axis, int64_t depth) const {
    throw std::runtime_error("FIXME: VirtualArray::localindex");
  }

  const ContentPtr
  VirtualArray::combinations(int64_t n,
                           bool replacement,
                           const util::RecordLookupPtr& recordlookup,
                           const util::Parameters& parameters,
                           int64_t axis,
                           int64_t depth) const {
    throw std::runtime_error("FIXME: VirtualArray::combinations");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceAt& at,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next(at)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceRange& range,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next(range)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceArray64& array,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next(array)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceField& field,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next(field)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceFields& fields,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next(fields)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceJagged64& jagged,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next(jagged)");
  }

  const ContentPtr
  VirtualArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceArray64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next_jagged(array)");
  }

  const ContentPtr
  VirtualArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceMissing64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next_jagged(missing)");
  }

  const ContentPtr
  VirtualArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceJagged64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_next_jagged(jagged)");
  }

}

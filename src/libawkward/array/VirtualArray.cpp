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
    throw std::runtime_error("FIXME: VirtualArray::tojson_part");
  }

  void
  VirtualArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    // Nothing to do: VirtualArrays contribute 0 to the nbytes.
  }

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
                                          cache_);
  }

  const ContentPtr
  VirtualArray::deep_copy(bool copyarrays,
                          bool copyindexes,
                          bool copyidentities) const {
    throw std::runtime_error("FIXME: VirtualArray::deep_copy");
  }

  void
  VirtualArray::check_for_iteration() const {
    throw std::runtime_error("FIXME: VirtualArray::check_for_iteration");
  }

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
    std::cout << "getitem_at_nowrap " << at << std::endl;

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
    std::cout << "getitem_range_nowrap " << start << " " << stop << std::endl;

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
    throw std::runtime_error("FIXME: VirtualArray::getitem_field");
  }

  const ContentPtr
  VirtualArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("FIXME: VirtualArray::getitem_fields");
  }

  const ContentPtr
  VirtualArray::carry(const Index64& carry) const {
    std::cout << "carry " << carry.tostring() << std::endl;

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
  VirtualArray::purelist_parameter(const std::string& key) const {
    throw std::runtime_error("FIXME: VirtualArray::purelist_parameter");
  }

  bool
  VirtualArray::purelist_isregular() const {
    throw std::runtime_error("FIXME: VirtualArray::purelist_isregular");
  }

  int64_t
  VirtualArray::purelist_depth() const {
    throw std::runtime_error("FIXME: VirtualArray::purelist_depth");
  }

  const std::pair<int64_t, int64_t>
  VirtualArray::minmax_depth() const {
    throw std::runtime_error("FIXME: VirtualArray::minmax_depth");
  }

  const std::pair<bool, int64_t>
  VirtualArray::branch_depth() const {
    throw std::runtime_error("FIXME: VirtualArray::branch_depth");
  }

  int64_t
  VirtualArray::numfields() const {
    throw std::runtime_error("FIXME: VirtualArray::numfields");
  }

  int64_t
  VirtualArray::fieldindex(const std::string& key) const {
    throw std::runtime_error("FIXME: VirtualArray::fieldindex");
  }

  const std::string
  VirtualArray::key(int64_t fieldindex) const {
    throw std::runtime_error("FIXME: VirtualArray::key");
  }

  bool
  VirtualArray::haskey(const std::string& key) const {
    throw std::runtime_error("FIXME: VirtualArray::haskey");
  }

  const std::vector<std::string>
  VirtualArray::keys() const {
    throw std::runtime_error("FIXME: VirtualArray::keys");
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

// // BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE
//
// #include <stdexcept>
//
// #include "awkward/Identity.h"
// #include "awkward/type/ListType.h"
//
// #include "awkward/fillable/ListFillable.h"
//
// namespace awkward {
//   int64_t ListFillable::length() const {
//     return index_.length();
//   }
//
//   void ListFillable::clear() {
//     index_.clear();
//     content_.get()->clear();
//   }
//
//   const std::shared_ptr<Type> ListFillable::type() const {
//     return std::shared_ptr<Type>(new ListType(content_.get()->type()));
//   }
//
//   const std::shared_ptr<Content> ListFillable::snapshot() const {
//     throw std::runtime_error("FIXME");
//   }
//
//   Fillable* ListFillable::null() {
//     index_.append(-1);
//     return this;
//   }
//
//   Fillable* ListFillable::boolean(bool x) {
//     int64_t length = content_.get()->length();
//     maybeupdate(content_.get()->boolean(x));
//     index_.append(length);
//     return this;
//   }
//
//   Fillable* ListFillable::integer(int64_t x) {
//     int64_t length = content_.get()->length();
//     maybeupdate(content_.get()->integer(x));
//     index_.append(length);
//     return this;
//   }
//
//   Fillable* ListFillable::real(double x) {
//     int64_t length = content_.get()->length();
//     maybeupdate(content_.get()->real(x));
//     index_.append(length);
//     return this;
//   }
//
//   void ListFillable::maybeupdate(Fillable* tmp) {
//     if (tmp != content_.get()) {
//       content_ = std::shared_ptr<Fillable>(tmp);
//     }
//   }
// }

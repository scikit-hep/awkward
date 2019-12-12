// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

// #ifndef AWKWARD_DRESSEDTYPE_H_
// #define AWKWARD_DRESSEDTYPE_H_
//
// #include <sstream>
//
// #include "awkward/type/Type.h"
//
// namespace awkward {
//   template <typename T>
//   class DressParameters {
//   public:
//     virtual const std::vector<std::string> keys() const = 0;
//     virtual const T get(const std::string& key) const = 0;
//     virtual const std::string get_string(const std::string& key) const = 0;
//     virtual bool equal(const DressParameters<T>& other, bool check_parameters) const = 0;
//   };
//
//   template <typename T>
//   class Dress {
//   public:
//     virtual const std::string name() const = 0;
//     virtual const std::string typestr(std::shared_ptr<Type> baretype, const DressParameters<T>& parameters) const = 0;
//     virtual bool equal(const Dress& other, bool check_parameters) const = 0;
//   };
//
//   template <typename D, typename P>
//   class DressedType: public Type {
//   public:
//     DressedType(const Parameters& parameters_FIXME, const std::shared_ptr<Type> type, const D& dress, const P& parameters): Type(parameters_FIXME), type_(type), dress_(dress), parameters_(parameters) { }
//
//     virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const {
//       std::string outstr = dress_.typestr(type_, parameters_);
//       if (outstr.size() != 0) {
//         return outstr;
//       }
//       else {
//         std::stringstream out;
//         out << indent << pre << "dress[" << type_.get()->tostring_part(indent, "", "") << ", " << util::quote(dress_.name(), false);
//         for (auto key : parameters_.keys()) {
//           out << ", " << key << "=" << parameters_.get_string(key);
//         }
//         out << "]";
//         return out.str();
//       }
//     }
//     virtual const std::shared_ptr<Type> shallow_copy() const {
//       return std::shared_ptr<Type>(new DressedType(parameters_FIXME_, type_, dress_, parameters_));
//     }
//     virtual bool equal(const std::shared_ptr<Type> other, bool check_parameters) const {
//       if (DressedType<D, P>* raw = dynamic_cast<DressedType<D, P>*>(other.get())) {
//         D otherdress = raw->dress();
//         if (!dress_.equal(otherdress, check_parameters)) {
//           return false;
//         }
//         P otherparam = raw->parameters();
//         if (!parameters_.equal(otherparam, check_parameters)) {
//           return false;
//         }
//         return true;
//       }
//       else {
//         return false;
//       }
//       return type_.get()->equal(dynamic_cast<DressedType<D, P>*>(other.get())->type(), check_parameters);
//     }
//     virtual std::shared_ptr<Type> level() const {
//       return type_.get()->level();
//     }
//     virtual std::shared_ptr<Type> inner() const {
//       return type_.get()->inner();
//     }
//     virtual std::shared_ptr<Type> inner(const std::string& key) const {
//       return type_.get()->inner(key);
//     }
//     virtual int64_t numfields() const {
//       return type_.get()->numfields();
//     }
//     virtual int64_t fieldindex(const std::string& key) const {
//       return type_.get()->fieldindex(key);
//     }
//     virtual const std::string key(int64_t fieldindex) const {
//       return type_.get()->key(fieldindex);
//     }
//     virtual bool haskey(const std::string& key) const {
//       return type_.get()->haskey(key);
//     }
//     virtual const std::vector<std::string> keyaliases(int64_t fieldindex) const {
//       return type_.get()->keyaliases(fieldindex);
//     }
//     virtual const std::vector<std::string> keyaliases(const std::string& key) const {
//       return type_.get()->keyaliases(key);
//     }
//     virtual const std::vector<std::string> keys() const {
//       return type_.get()->keys();
//     }
//
//     const std::shared_ptr<Type> type() const { return type_; };
//     const D dress() const { return dress_; };
//     const P parameters() const { return parameters_; }
//
//   private:
//     const std::shared_ptr<Type> type_;
//     const D dress_;
//     const P parameters_;
//   };
// }
//
// #endif // AWKWARD_DRESSEDTYPE_H_

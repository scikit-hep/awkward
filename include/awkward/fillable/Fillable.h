// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_FILLABLE_H_
#define AWKWARD_FILLABLE_H_

#include "awkward/cpu-kernels/util.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"

namespace awkward {
  class Fillable {
  public:
    virtual int64_t length() const = 0;
    virtual void clear() = 0;
    virtual const std::shared_ptr<Type> type() const = 0;
    virtual const std::shared_ptr<Content> tolayout() = 0;

    virtual Fillable* null() = 0;
    virtual Fillable* boolean(bool x) = 0;
  };

  template<typename T>
  class vector_deleter {
  public:
    vector_deleter(std::vector<T>* vector) {
      vector_ = new std::vector<T>();
      vector_->swap(*vector);
    }

    void operator()(T const *p) {
      vector_->clear();
      delete vector_;
    }

  private:
    std::vector<T>* vector_;
  };

  // template<typename T>
  // class TransferOwnership {
  // public:
  //   TransferOwnership(std::vector<T>& vector): vector_(new std::vector<T>()) {
  //     vector_->swap(vector);
  //   }
  //
  //   TransferOwnership(const TransferOwnership& ownership) {
  //     vector_ = ownership.vector_;
  //   }
  //
  //   void operator()(T const *p) {
  //     vector_->clear();
  //     delete vector_;
  //   }
  //
  //   const T* rawptr() const {
  //     return vector_->data();
  //   }
  //
  // private:
  //   std::vector<T>* vector_;
  // };
  //
  // template<typename T>
  // std::shared_ptr<T> vector_to_sharedptr(std::vector<T>& vector) {
  //   TransferOwnership<T> ownership(vector);
  //   return std::shared_ptr<T>(ownership.rawptr(), TransferOwnership<T>(ownership));
  // }
}

#endif // AWKWARD_FILLABLE_H_

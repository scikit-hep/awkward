// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <vector>
#include <memory>
#include <numeric>
#include <cmath>

namespace awkward {
  /// @class GrowableBuffer
  ///
  /// @brief Discontiguous, one-dimensional buffer which consists of
  /// multiple contiguous, one-dimensional arrays that can grow
  /// indefinitely by calling #append.
  ///
  /// The buffer starts by reserving #initial number of slots. When the
  /// number of slots used reaches the number reserved, a new panel is
  /// allocated that has a size equal to #initial.
  ///
  /// When {@link ArrayBuilder#snapshot ArrayBuilder::snapshot} is called,
  /// these buffers are copied to the new Content array.
  template <typename PRIMITIVE>
  class GrowableBuffer {
  public:
    /// @brief Creates an empty GrowableBuffer.
    ///
    /// @param initial Initial size configuration for building a panel.
    static GrowableBuffer<PRIMITIVE>
    empty(size_t initial) {
      return empty(initial, 0);
    }

    /// @brief Creates an empty GrowableBuffer with a minimum reservation.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param minreserve The initial reservation will be the maximum
    /// of `minreserve` and #initial.
    static GrowableBuffer<PRIMITIVE>
    empty(size_t initial, size_t minreserve) {
      size_t actual = initial;
      if (actual < minreserve) {
        actual = minreserve;
      }
      return GrowableBuffer(initial,
        std::unique_ptr<PRIMITIVE>(new PRIMITIVE[actual]),
        0, actual);
    }

    /// @brief Creates a GrowableBuffer in which all elements are initialized to `0`.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html).
    static GrowableBuffer<PRIMITIVE>
    zeros(size_t initial, size_t length) {
      size_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (size_t i = 0;  i < length;  i++) {
        rawptr[i] = 0;
      }
      return GrowableBuffer(initial, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer in which all elements are initialized
    /// to a given value.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param value The initialization value.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [full](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html).
    static GrowableBuffer<PRIMITIVE>
    full(size_t initial, PRIMITIVE value, size_t length) {
      size_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (size_t i = 0;  i < length;  i++) {
        rawptr[i] = value;
      }
      return GrowableBuffer<PRIMITIVE>(initial, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer in which the elements are initialized
    /// to numbers counting from `0` to `length`.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).
    static GrowableBuffer<PRIMITIVE>
    arange(size_t initial, size_t length) {
      size_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (size_t i = 0;  i < length;  i++) {
        rawptr[i] = (PRIMITIVE)i;
      }
      return GrowableBuffer(initial, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer from a full set of parameters.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param length Currently used number of elements.
    /// @param reserved Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    GrowableBuffer(size_t initial,
                   std::unique_ptr<PRIMITIVE> ptr,
                   size_t length,
                   size_t reserved)
        : initial_(initial) {
      ptr_.push_back(std::move(ptr));
      length_.push_back(length);
      reserved_.push_back(reserved);
    }

    /// @brief Creates a GrowableBuffer by allocating a new buffer, taking an
    /// initial #reserved from #initial.
    GrowableBuffer(size_t initial)
        : GrowableBuffer(initial,
                         std::unique_ptr<PRIMITIVE>(new PRIMITIVE[initial]),
                         0,
                         initial) { }

    /// @brief Currently used number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    size_t
    length() const {
      return std::accumulate(length_.begin(), length_.end(), (size_t)0);
    }

    /// @brief Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    size_t
    reserved() const {
      return std::accumulate(reserved_.begin(), reserved_.end(), (size_t)0);
    }

    /// @brief Discards accumulated data, the #reserved returns to
    /// initial, and a new #ptr is allocated.
    void
    clear() {
      length_.clear();
      length_.push_back(0);
      reserved_.clear();
      reserved_.push_back(initial_);
      ptr_.clear();
      ptr_.push_back(std::unique_ptr<PRIMITIVE>(new PRIMITIVE[initial_]));
    }

    /// @brief Inserts one `datum` into the panel, possibly triggering
    /// allocation of a new panel.
    ///
    /// This increases the #length by 1; if the new #length is larger than
    /// #reserved, a new panel will be allocated.
    void
    append(PRIMITIVE datum) {
      if (length_[ptr_.size()-1] == reserved_[ptr_.size()-1]) {
        add_panel(reserved_[ptr_.size()-1]);
      }
      fill_panel(datum);
    }

    /// @brief Inserts an entire array into the panel(s), possibly triggering
    /// allocation of a new panel.
    ///
    /// If the size is larger than the empty slots in the current panel, then,
    /// first, the empty slots are filled and then a new panel will be allocated
    /// for the rest of the array elements.
    void
    extend(PRIMITIVE* ptr, size_t size) {
      size_t empty_slots = reserved_[ptr_.size()-1] - length_[ptr_.size()-1];
      if (size > empty_slots) {
        for (size_t i = 0; i < empty_slots; i++) {
          fill_panel(ptr[i]);
        }
        add_panel(size - empty_slots > reserved_[ptr_.size()-1] ?
                  size - empty_slots : reserved_[ptr_.size()-1]);
        for (size_t i = empty_slots; i < size; i++) {
          fill_panel(ptr[i]);
        }
      }
      else {
        for (size_t i = 0; i < size; i++) {
          fill_panel(ptr[i]);
        }
      }
    }

    /// @brief Like append, but the type signature returns the reference to `PRIMITIVE`.
    PRIMITIVE& append_and_get_ref(PRIMITIVE datum) {
      append(datum);
      return (&*ptr_.back())[length_.back()];
    }

    /// @brief Copies and concatenates all accumulated data from multiple panels to one
    /// contiguously allocated `external_pointer`.
    void
    concatenate(PRIMITIVE* external_pointer) const noexcept {
      size_t next_panel = 0;
      for (size_t i = 0;  i < ptr_.size();  i++) {
        memcpy(external_pointer + next_panel, reinterpret_cast<void*>(ptr_[i].get()), length_[i]*sizeof(PRIMITIVE));
        next_panel += length_[i];
      }
    }

    /// @brief Checks whether the array is contiguous.
    size_t is_contiguous() {
      return (ptr_.size() == 1);
    }

    /// @brief Takes this (possibly multi-panels) GrowableBuffer<PRIMITIVE>
    /// and makes another (one panel) GrowableBuffer<TO_PRIMITIVE>.
    ///
    /// Used to change the data type of buffer content from `PRIMITIVE`
    /// to `TO_PRIMITIVE` for building arrays.
    template<typename TO_PRIMITIVE>
    GrowableBuffer<TO_PRIMITIVE>
    copy_as() {
      auto len = length();
      auto ptr = std::unique_ptr<TO_PRIMITIVE>(new TO_PRIMITIVE[len]);
      TO_PRIMITIVE* rawptr = ptr.get();
      int64_t k = 0;
      for (size_t i = 0;  i < ptr_.size(); i++) {
        for (int64_t j = 0; j < length_[i]; j++) {
          rawptr[k] = static_cast<TO_PRIMITIVE>(ptr_[i].get()[j]);
          k++;
        }
      }
      return GrowableBuffer<TO_PRIMITIVE>(len, std::move(ptr), len, len);
    }
  private:
    /// @brief Inserts one `datum` into the panel.
    void
    fill_panel(PRIMITIVE datum) {
        ptr_[ptr_.size()-1].get()[length_[ptr_.size()-1]] = datum;
        length_[ptr_.size()-1]++;
    }

    /// @brief Creates a new panel with slots equal to #reserved.
    void
    add_panel(size_t reserved) {
      ptr_.push_back(std::unique_ptr<PRIMITIVE>(new PRIMITIVE[reserved]));
      length_.push_back(0);
      reserved_.push_back(reserved);
    }

    /// @brief Initial size configuration for building a panel.
    size_t initial_;

    /// @brief Vector of unique pointers to the panels.
    std::vector<std::unique_ptr<PRIMITIVE>> ptr_;

    /// @brief Vector containing the lengths of the panels.
    ///
    /// Each index of this vector is aligned with the index of the
    /// vector of unique pointers to the panels.
    std::vector<size_t> length_;

    /// @brief Vector containing the reserved sizes of the panels.
    ///
    /// Each index of this vector is aligned with the index of the
    /// vector of unique pointers to the panels.
    std::vector<size_t> reserved_;
  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_

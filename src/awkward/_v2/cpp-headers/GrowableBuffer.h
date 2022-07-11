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
    empty(int64_t initial) {
      return empty(initial, 0);
    }

    /// @brief Creates an empty GrowableBuffer with a minimum reservation.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param minreserve The initial reservation will be the maximum
    /// of `minreserve` and #initial.
    static GrowableBuffer<PRIMITIVE>
    empty(int64_t initial, int64_t minreserve) {
      int64_t actual = initial;
      if (actual < minreserve) {
        actual = minreserve;
      }
      return GrowableBuffer(initial,
        std::unique_ptr<PRIMITIVE>(new PRIMITIVE[(size_t)actual]),
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
    zeros(int64_t initial, int64_t length) {
      int64_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[(size_t)actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0;  i < length;  i++) {
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
    full(int64_t initial, PRIMITIVE value, int64_t length) {
      int64_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[(size_t)actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0;  i < length;  i++) {
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
    arange(int64_t initial, int64_t length) {
      int64_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE>(new PRIMITIVE[(size_t)actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0;  i < length;  i++) {
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
    GrowableBuffer(int64_t initial,
                   std::unique_ptr<PRIMITIVE> ptr,
                   int64_t length,
                   int64_t reserved)
        : initial_(initial),
          total_length_(length),
          current_length_((size_t)length),
          current_reserved_((size_t)reserved) {
      ptr_.emplace_back(std::move(ptr));
      reserved_.emplace_back(reserved);

      ptr_.reserve(1024);
      length_.reserve(1024);
      reserved_.reserve(1024);
    }

    /// @brief Creates a GrowableBuffer by allocating a new buffer, taking an
    /// initial #reserved from #initial.
    GrowableBuffer(int64_t initial)
        : GrowableBuffer(initial,
                         std::unique_ptr<PRIMITIVE>(new PRIMITIVE[initial]),
                         0,
                         initial) { }

    /// @brief Currently used number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    int64_t
    length() const {
      return (is_contiguous() ? (int64_t)current_length_ : total_length_ + (int64_t)current_length_);
    }

    /// @brief Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    int64_t
    reserved() const {
      return std::accumulate(reserved_.begin(), reserved_.end(), (int64_t)0);
    }

    /// @brief Discards accumulated data, the #reserved returns to
    /// initial, and a new #ptr is allocated.
    void
    clear() {
      length_.clear();
      reserved_.clear();
      reserved_.emplace_back(initial_);

      total_length_ = 0;
      current_length_ = 0;
      current_reserved_ = (size_t)initial_;
    }

    void
    reset() {
      length_.clear();
      reserved_.clear();
      ptr_.clear();

      total_length_ = 0;
      current_length_ = 0;
      current_reserved_ = (size_t)initial_;
    }

    /// @brief Inserts one `datum` into the panel, possibly triggering
    /// allocation of a new panel.
    ///
    /// This increases the #length by 1; if the new #length is larger than
    /// #reserved, a new panel will be allocated.
    void
    append(PRIMITIVE datum) {
      if (current_length_ == current_reserved_) {
        add_panel(current_reserved_ << 1);
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
      size_t empty_slots = current_reserved_ - current_length_;
      if (size > empty_slots) {
        for (size_t i = 0; i < empty_slots; i++) {
          fill_panel(ptr[i]);
        }
        add_panel(size - empty_slots > current_reserved_ ?
                  size - empty_slots : current_reserved_);
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
      return (&*ptr_.back())[current_length_];
    }

    /// @brief Copies and concatenates all accumulated data from multiple panels to one
    /// contiguously allocated `external_pointer`.
    void
    concatenate(PRIMITIVE* external_pointer) const noexcept {
      if (!is_empty()) {
        int64_t next_panel = 0;
        size_t i = 0;
        size_t num_full_panels = ptr_.size() - 1;
        for ( ;  i < num_full_panels;  i++) {
          memcpy(external_pointer + next_panel, reinterpret_cast<void*>(ptr_[i].get()), length_[i] * sizeof(PRIMITIVE));
          next_panel += length_[i];
        }
        // and the last panel
        memcpy(external_pointer + next_panel, reinterpret_cast<void*>(ptr_[i].get()), current_length_ * sizeof(PRIMITIVE));
      }
    }

    /// @brief Checks whether the array is contiguous.
    bool is_contiguous() const {
      return (ptr_.size() == 1);
    }

    /// @brief Checks whether the GowableBuffer has any panels.
    bool is_empty() const {
      return (ptr_.size() == 0);
    }

    /// @brief Takes this (possibly multi-panels) GrowableBuffer<PRIMITIVE>
    /// and makes another (one panel) GrowableBuffer<TO_PRIMITIVE>.
    ///
    /// Used to change the data type of buffer content from `PRIMITIVE`
    /// to `TO_PRIMITIVE` for building arrays.
    template<typename TO_PRIMITIVE>
    GrowableBuffer<TO_PRIMITIVE>
    copy_as() {
      size_t num_full_panels = ptr_.size() - 1;
      auto len = length();
      int64_t actual = (len < initial_) ? initial_ : len;

      auto ptr = std::unique_ptr<TO_PRIMITIVE>(new TO_PRIMITIVE[(size_t)actual]);
      TO_PRIMITIVE* rawptr = ptr.get();
      int64_t k = 0;
      size_t i = 0;
      for ( ;  i < num_full_panels; i++) {
        for (size_t j = 0; j < length_[i]; j++) {
          rawptr[k] = static_cast<TO_PRIMITIVE>(ptr_[i].get()[j]);
          k++;
        }
      }
      // and the last one
      for (size_t j = 0; j < current_length_; j++) {
        rawptr[k] = static_cast<TO_PRIMITIVE>(ptr_[i].get()[j]);
        k++;
      }

      return GrowableBuffer<TO_PRIMITIVE>(actual, std::move(ptr), len, actual);
    }

  private:
    /// @brief Inserts one `datum` into the panel.
    void
    fill_panel(PRIMITIVE datum) {
        (&*ptr_.back())[current_length_++] = datum;
    }

    /// @brief Creates a new panel with slots equal to #reserved.
    void
    add_panel(size_t reserved) {
      ptr_.emplace_back(std::unique_ptr<PRIMITIVE>(new PRIMITIVE[reserved]));
      length_.emplace_back(current_length_);
      reserved_.emplace_back(reserved);
      total_length_ += current_length_;
      current_length_ = 0;
      current_reserved_ = reserved;
    }

    /// @brief Initial size configuration for building a panel.
    int64_t initial_;

    /// @brief Vector of unique pointers to the panels.
    std::vector<std::unique_ptr<PRIMITIVE>> ptr_;

    /// @brief Vector containing the lengths of the filled panels.
    ///
    /// Each index of this vector is aligned with the index of the
    /// vector of unique pointers to the filled panels.
    std::vector<size_t> length_;

    /// @brief Total length of data in all panels including an unfilled one.
    int64_t total_length_;

    /// @brief Current length of an unfilled panel.
    size_t current_length_;

    /// @brief Vector containing the reserved sizes of the panels.
    ///
    /// Each index of this vector is aligned with the index of the
    /// vector of unique pointers to the panels.
    std::vector<size_t> reserved_;

    /// @brief Reserved size of a current panel.
    size_t current_reserved_;

  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_

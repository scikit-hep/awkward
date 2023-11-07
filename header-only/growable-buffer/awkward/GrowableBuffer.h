// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include "awkward/BuilderOptions.h"

#include <cstring>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>
#include <complex>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <stdint.h>

namespace awkward {

  template <template <class...> class TT, class... Args>
  std::true_type is_tt_impl(TT<Args...>);
  template <template <class...> class TT>
  std::false_type is_tt_impl(...);

  template <template <class...> class TT, class T>
  using is_tt = decltype(is_tt_impl<TT>(std::declval<typename std::decay<T>::type>()));

  template <typename PRIMITIVE>
  /// @class Panel
  ///
  /// Creates a contiguous, one-dimensional panel.
  class Panel {
  public:

    /// @brief Creates a Panel by allocating a new panel, taking a
    /// #reserved number of slots.
    ///
    /// @param reserved Currently allocated number of elements in the panel.
    Panel(size_t reserved)
        : ptr_(std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[reserved])),
          length_(0),
          reserved_(reserved),
          next_(nullptr) {}

    /// @brief Creates a Panel from a full set of parameters.
    ///
    /// @param ptr Unique reference to the panel data.
    /// @param length Currently number used of elements in the panel.
    /// @param reserved Currently allocated number of elements in the panel.
    Panel(std::unique_ptr<PRIMITIVE[]> ptr, size_t length, size_t reserved)
        : ptr_(std::move(ptr)),
          length_(length),
          reserved_(reserved),
          next_(nullptr) {}

    /// @brief Deletes a Panel.
    ///
    /// Unchain the pointers to avoid a stack overflow when
    /// a recursive implicit destructor is invoked.
    ~Panel() {
      for (std::unique_ptr<Panel> current = std::move(next_);
           current;
           current = std::move(current->next_));
    }

    /// @brief Overloads [] operator to access elements like an array.
    PRIMITIVE& operator[](size_t i) { return ptr_.get()[i]; }

    /// @brief Creates a new panel with slots equal to #reserved and
    /// appends it after the current panel.
    Panel*
    append_panel(size_t reserved) {
      next_ = std::move(std::unique_ptr<Panel>(new Panel(reserved)));
      return next_.get();
    }

    /// @brief Inserts one `datum` into the panel.
    void
    fill_panel(PRIMITIVE datum) {
      ptr_.get()[length_++] = datum;
    }

    /// @brief Pointer to the next panel.
    std::unique_ptr<Panel>&
    next() {
      return next_;
    }

    /// @brief Currently used number of elements in the panel.
    size_t
    current_length() {
      return length_;
    }

    /// @brief Currently allocated number of elements in the panel.
    size_t
    reserved() {
      return reserved_;
    }

    /// @brief Unique pointer to the panel data.
    std::unique_ptr<PRIMITIVE[]>&
    data() {
      return ptr_;
    }

    /// @brief Copies the data from a panel to one contiguously allocated `to_ptr`.
    ///
    /// @param to_ptr One contiguously allocated panel.
    /// @param offset Distance between `to_ptr` and the pointer to the destination where the
    /// accumulated data is copied.
    /// @param from Distance between `ptr` and pointer to the source of the data to be copied.
    /// @param length Length of the data to be copied.
    void
    append(PRIMITIVE* to_ptr, size_t offset, size_t from, int64_t length) const noexcept {
      memcpy(to_ptr + offset,
             reinterpret_cast<void*>(ptr_.get() + from),
             length * sizeof(PRIMITIVE) - from);
    }

    /// @brief Copies and concatenates the accumulated data from multiple panels `ptr_` to one
    /// contiguously allocated `to_ptr`.
    ///
    /// @param to_ptr One contiguously allocated panel.
    /// @param offset Distance between `to_ptr` and the pointer to the destination where the
    /// accumulated data is copied.
    /// @param from Distance between `ptr` and pointer to the source of the data to be copied.
    void
    concatenate_to_from(PRIMITIVE* to_ptr, size_t offset, size_t from) const noexcept {
      memcpy(to_ptr + offset,
             reinterpret_cast<void*>(ptr_.get() + from),
             length_ * sizeof(PRIMITIVE) - from);
      if (next_) {
        next_->concatenate_to(to_ptr, offset + length_);
      }
    }

    /// @brief Copies and concatenates the accumulated data from multiple panels `ptr_` to one
    /// contiguously allocated `to_ptr`.
    ///
    /// @param to_ptr One contiguously allocated panel.
    /// @param offset Distance between `to_ptr` and the pointer to the destination where the
    /// accumulated data is copied.
    void
    concatenate_to(PRIMITIVE* to_ptr, size_t offset) const noexcept {
      memcpy(to_ptr + offset,
             reinterpret_cast<void*>(ptr_.get()),
             length_ * sizeof(PRIMITIVE));
      if (next_) {
        next_->concatenate_to(to_ptr, offset + length_);
      }
    }

    /// @brief Fills (one panel) GrowableBuffer<TO_PRIMITIVE> with the
    /// elements of (possibly multi-panels) GrowableBuffer<PRIMITIVE>.
    ///
    /// Changes the data type from `PRIMITIVE` to `TO_PRIMITIVE`/
    template <typename TO_PRIMITIVE>
    typename std::enable_if<(!awkward::is_tt<std::complex, TO_PRIMITIVE>::value &&
                             !awkward::is_tt<std::complex, PRIMITIVE>::value) ||
                            (awkward::is_tt<std::complex, TO_PRIMITIVE>::value &&
                             awkward::is_tt<std::complex, PRIMITIVE>::value)>::type
    copy_as(TO_PRIMITIVE* to_ptr, size_t offset) {
      for (size_t i = 0; i < length_; i++) {
        to_ptr[offset++] = static_cast<TO_PRIMITIVE>(ptr_.get()[i]);
      }
      if (next_) {
        next_->copy_as(to_ptr, offset);
      }
    }

    template <typename TO_PRIMITIVE>
    typename std::enable_if<!awkward::is_tt<std::complex, TO_PRIMITIVE>::value &&
                             awkward::is_tt<std::complex, PRIMITIVE>::value>::type
    copy_as(TO_PRIMITIVE* to_ptr, size_t offset) {
      for (size_t i = 0; i < length_; i++) {
        to_ptr[offset++] = static_cast<TO_PRIMITIVE>(ptr_.get()[i].real());
        to_ptr[offset++] = static_cast<TO_PRIMITIVE>(ptr_.get()[i].imag());
      }
      if (next_) {
        next_->copy_as(to_ptr, offset);
      }
    }

    /// @brief 'copy_as' specialization of a 'std::complex' template type.
    /// Fills (one panel) GrowableBuffer<std::complex> with the
    /// elements of (possibly multi-panels) GrowableBuffer<PRIMITIVE>.
    ///
    /// Changes the data type from `PRIMITIVE` to `std::complex`/
    template <typename TO_PRIMITIVE>
    typename std::enable_if<awkward::is_tt<std::complex, TO_PRIMITIVE>::value &&
                            !awkward::is_tt<std::complex, PRIMITIVE>::value>::type
    copy_as(TO_PRIMITIVE* to_ptr, size_t offset) {
      for (size_t i = 0; i < length_; i++) {
        double val = static_cast<double>(ptr_.get()[i]);
        to_ptr[offset++] = TO_PRIMITIVE(val);
      }
      if (next_) {
        next_->copy_as(to_ptr, offset);
      }
    }

  private:
    /// @brief Unique pointer to the panel data.
    std::unique_ptr<PRIMITIVE[]> ptr_;

    /// @brief The length of the panel data.
    size_t length_;

    /// @brief Reserved size of the panel.
    size_t reserved_;

    /// @brief Pointer to the next Panel.
    std::unique_ptr<Panel> next_;
  };

  /// @class GrowableBuffer
  ///
  /// @brief Discontiguous, one-dimensional buffer (which consists of
  /// multiple contiguous, one-dimensional panels) that can grow
  /// indefinitely by calling #append.
  ///
  /// Configured by BuilderOptions, the buffer starts by reserving
  /// {@link BuilderOptions#initial initial} slots.
  /// When the number of slots used reaches the number reserved, a new
  /// panel is allocated that is
  /// {@link BuilderOptions#resize resize} times larger.
  ///
  /// When {@link ArrayBuilder#to_buffers ArrayBuilder::to_buffers} is called,
  /// these buffers are copied to the new Content array.
  template <typename PRIMITIVE>
  class GrowableBuffer {
  public:
    /// @brief Creates an empty GrowableBuffer.
    ///
    /// @param options Initial size configuration for building a panel.
    static GrowableBuffer<PRIMITIVE>
    empty(const BuilderOptions& options) {
      return empty(options, 0);
    }

    /// @brief Creates an empty GrowableBuffer with a minimum reservation.
    ///
    /// @param options Initial size configuration for building a panel.
    /// @param minreserve The initial reservation will be the maximum
    /// of `minreserve` and
    /// {@link BuilderOptions#initial initial}.
    static GrowableBuffer<PRIMITIVE>
    empty(const BuilderOptions& options, int64_t minreserve) {
      int64_t actual = options.initial();
      if (actual < minreserve) {
        actual = minreserve;
      }
      return GrowableBuffer(
          options,
          std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[(size_t)actual]),
          0,
          actual);
    }

    /// @brief Creates a GrowableBuffer in which all elements are initialized to `0`.
    ///
    /// @param options Initial size configuration for building a panel.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html).
    static GrowableBuffer<PRIMITIVE>
    zeros(const BuilderOptions& options, int64_t length) {
      int64_t actual = options.initial();
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[(size_t)actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0; i < length; i++) {
        rawptr[i] = 0;
      }
      return GrowableBuffer(options, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer in which all elements are initialized
    /// to a given value.
    ///
    /// @param options Initial size configuration for building a panel.
    /// @param value The initialization value.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [full](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html).
    static GrowableBuffer<PRIMITIVE>
    full(const BuilderOptions& options, PRIMITIVE value, int64_t length) {
      int64_t actual = options.initial();
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[(size_t)actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0; i < length; i++) {
        rawptr[i] = value;
      }
      return GrowableBuffer<PRIMITIVE>(options, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer in which the elements are initialized
    /// to numbers counting from `0` to `length`.
    ///
    /// @param options Initial size configuration for building a panel.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).
    static GrowableBuffer<PRIMITIVE>
    arange(const BuilderOptions& options, int64_t length) {
      int64_t actual = options.initial();
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[(size_t)actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0; i < length; i++) {
        rawptr[i] = (PRIMITIVE)i;
      }
      return GrowableBuffer(options, std::move(ptr), length, actual);
    }

    /// @brief Takes a (possibly multi-panels) GrowableBuffer<PRIMITIVE>
    /// and makes another (one panel) GrowableBuffer<TO_PRIMITIVE>.
    ///
    /// Used to change the data type of buffer content from `PRIMITIVE`
    /// to `TO_PRIMITIVE` for building arrays.
    template <typename TO_PRIMITIVE>
    static GrowableBuffer<TO_PRIMITIVE>
    copy_as(const GrowableBuffer<PRIMITIVE>& other) {
      int64_t len = (int64_t)other.length();
      int64_t actual =
          (len < other.options_.initial()) ? other.options_.initial() : len;

      if (!awkward::is_tt<std::complex, TO_PRIMITIVE>::value &&
        awkward::is_tt<std::complex, PRIMITIVE>::value) {
          len *= 2;
          actual *= 2;
        }

      auto ptr =
          std::unique_ptr<TO_PRIMITIVE[]>(new TO_PRIMITIVE[(size_t)actual]);
      TO_PRIMITIVE* rawptr = ptr.get();

      other.panel_->copy_as(rawptr, 0);

      return GrowableBuffer<TO_PRIMITIVE>(
          BuilderOptions(actual, other.options().resize()),
          std::move(ptr),
          len,
          actual);
    }

    /// @brief Creates a GrowableBuffer from a full set of parameters.
    ///
    /// @param options Initial size configuration for building a panel.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param length Currently used number of elements.
    /// @param reserved Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    GrowableBuffer(const BuilderOptions& options,
                   std::unique_ptr<PRIMITIVE[]> ptr,
                   int64_t length,
                   int64_t reserved)
        : options_(options),
          length_(0),
          panel_(std::unique_ptr<Panel<PRIMITIVE>>(new Panel<PRIMITIVE>(
              std::move(ptr), (size_t)length, (size_t)reserved))),
          ptr_(panel_.get()) {}

    /// @brief Creates a GrowableBuffer by allocating a new buffer, taking an
    /// options #reserved from #options.
    ///
    /// @param options Initial size configuration for building a panel.
    GrowableBuffer(const BuilderOptions& options)
        : GrowableBuffer(options,
                         std::unique_ptr<PRIMITIVE[]>(
                             new PRIMITIVE[(size_t)options.initial()]),
                         0,
                         options.initial()) {}

    /// @brief Move constructor
    ///
    /// panel_ is move-only.
    GrowableBuffer(GrowableBuffer&& other) noexcept
        : options_(other.options_),
          length_(other.length_),
          panel_(std::move(other.panel_)),
          ptr_(other.ptr_) {}

    /// @brief Currently used number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    size_t
    length() const {
      return length_ + ptr_->current_length();
    }

    /// @brief Return options of this GrowableBuffer.
    const BuilderOptions&
    options() const {
      return options_;
    }

    /// @brief Discards accumulated data, the #reserved returns to
    /// options.initial(), and a new #ptr is allocated.
    void
    clear() {
      panel_ = std::move(std::unique_ptr<Panel<PRIMITIVE>>(
          new Panel<PRIMITIVE>((size_t)options_.initial())));
      ptr_ = panel_.get();
      length_ = 0;
    }

    /// @brief Last element in last panel
    PRIMITIVE
    last() const {
      if (ptr_->current_length() == 0) {
        throw std::runtime_error("Buffer is empty");
      } else {
        return (*ptr_)[ptr_->current_length() - 1];
      }
    }

    /// @brief Currently used number of bytes.
    size_t
    nbytes() const {
      return length() * sizeof(PRIMITIVE);
    }

    /// @brief Inserts one `datum` into the panel, possibly triggering
    /// allocation of a new panel.
    ///
    /// This increases the #length by 1; if the new #length is larger than
    /// #reserved, a new panel will be allocated.
    void
    append(PRIMITIVE datum) {
      if (ptr_->current_length() == ptr_->reserved()) {
        add_panel((size_t)ceil(options_.initial() * options_.resize()));
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
    extend(const PRIMITIVE* ptr, size_t size) {
      size_t unfilled_items = ptr_->reserved() - ptr_->current_length();
      if (size > unfilled_items) {
        for (size_t i = 0; i < unfilled_items; i++) {
          fill_panel(ptr[i]);
        }
        add_panel(size - unfilled_items > ptr_->reserved() ? size - unfilled_items
                                                        : ptr_->reserved());
        for (size_t i = unfilled_items; i < size; i++) {
          fill_panel(ptr[i]);
        }
      } else {
        for (size_t i = 0; i < size; i++) {
          fill_panel(ptr[i]);
        }
      }
    }

    /// @brief Like append, but the type signature returns the reference to `PRIMITIVE`.
    PRIMITIVE&
    append_and_get_ref(PRIMITIVE datum) {
      append(datum);
      return (*ptr_)[ptr_->current_length() - 1];
    }

    /// @brief Copies and concatenates all accumulated data from multiple panels to one
    /// contiguously allocated `external_pointer`.
    void
    concatenate(PRIMITIVE* external_pointer) const noexcept {
      if (external_pointer) {
        panel_->concatenate_to(external_pointer, 0);
      }
    }

    /// @brief Moves all accumulated data from multiple panels to one
    /// contiguously allocated `external_pointer`. The panels are deleted,
    /// and a new #ptr is allocated.
    void
    move_to(PRIMITIVE* to_ptr) noexcept {
      size_t next_offset = 0;
      while(panel_) {
        memcpy(to_ptr + next_offset,
               reinterpret_cast<void*>(panel_.get()->data().get()),
               panel_.get()->current_length() * sizeof(PRIMITIVE));
        next_offset += panel_.get()->current_length();
        panel_ = std::move(panel_.get()->next());
      }
      clear();
    }

    /// @brief Copies and concatenates all accumulated data from multiple panels to one
    /// contiguously allocated `external_pointer`.
    void
    concatenate_from(PRIMITIVE* external_pointer, size_t to, size_t from) const noexcept {
      if (external_pointer) {
        panel_->concatenate_to_from(external_pointer, to, from);
      }
    }

    /// @brief Copies data from a panel to one contiguously allocated `external_pointer`.
    void
    append(PRIMITIVE* external_pointer, size_t offset, size_t from, int64_t length) const noexcept {
      if (external_pointer) {
        panel_->append(external_pointer, offset, from, length);
      }
    }

  private:
    /// @brief Fills the data into the panel one by one.
    void
    fill_panel(PRIMITIVE datum) {
      ptr_->fill_panel(datum);
    }

    /// @brief Adds a new panel with slots equal to #reserved.
    /// and updates the current panel pointer to it.
    void
    add_panel(size_t reserved) {
      length_ += ptr_->current_length();
      ptr_ = ptr_->append_panel(reserved);
    }

    /// @brief Initial size configuration for building a panel.
    const BuilderOptions options_;

    /// @brief Filled panels data length.
    size_t length_;

    /// @brief The first panel.
    std::unique_ptr<Panel<PRIMITIVE>> panel_;

    /// @brief A pointer to a current panel.
    ///
    /// Points to the address of the first byte of the current panel.
    Panel<PRIMITIVE>* ptr_;
  };
}  // namespace awkward

#endif  // AWKWARD_GROWABLEBUFFER_H_

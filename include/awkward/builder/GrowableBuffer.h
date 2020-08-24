// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <cmath>
#include <cstring>

#include "awkward/common.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/Index.h"

namespace awkward {
  /// @class GrowableBuffer
  ///
  /// @brief Contiguous, one-dimensional array that can grow indefinitely
  /// by calling #append.
  ///
  /// Configured by ArrayBuilderOptions, the buffer starts by reserving
  /// {@link ArrayBuilderOptions#initial ArrayBuilderOptions::initial} slots.
  /// When the number of slots used reaches the number reserved, a new
  /// buffer is allocated that is
  /// {@link ArrayBuilderOptions#resize ArrayBuilderOptions::resize} times
  /// larger. Thus, a logarithmic number of reallocations are needed as
  /// data grow.
  ///
  /// When {@link ArrayBuilder#snapshot ArrayBuilder::snapshot} is called,
  /// these buffers are shared with the new Content array. The GrowableBuffer
  /// can still grow because the Content array only sees the part of its
  /// reservation that existed at the time of the snapshot (new elements are
  /// beyond its {@link Content#length Content::length}).
  ///
  /// If a GrowableBuffer resizes itself by allocating a new array, it
  /// decreases the reference counter for the shared buffer, but the Content
  /// still owns it, and thus becomes the sole owner.
  ///
  /// The only disadvantage to this scheme is that the Content might forever
  /// have a reservation that is larger than it needs and it is unable to
  /// delete or take advantage of. However, many operations require buffers
  /// to be rewritten; under normal circumstances, it would soon be replaced
  /// by a more appropriately sized buffer.
  template <typename T>
  class LIBAWKWARD_EXPORT_SYMBOL GrowableBuffer {
  public:
    /// @brief Creates an empty GrowableBuffer.
    ///
    /// @param options Configuration options for building an array.
    static GrowableBuffer<T>
      empty(const ArrayBuilderOptions& options);

    /// @brief Creates an empty GrowableBuffer with a minimum reservation.
    ///
    /// @param options Configuration options for building an array.
    /// @param minreserve The initial reservation will be the maximum
    /// of `minreserve` and
    /// {@link ArrayBuilderOptions#initial ArrayBuilderOptions::initial}.
    static GrowableBuffer<T>
      empty(const ArrayBuilderOptions& options, int64_t minreserve);

    /// @brief Creates a GrowableBuffer in which all elements are initialized
    /// to a given value.
    ///
    /// @param options Configuration options for building an array.
    /// @param value The initialization value.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [full](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html).
    static GrowableBuffer<T>
      full(const ArrayBuilderOptions& options, T value, int64_t length);

    /// @brief Creates a GrowableBuffer in which the elements are initialized
    /// to numbers counting from `0` to `length`.
    ///
    /// @param options Configuration options for building an array.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).
    static GrowableBuffer<T>
      arange(const ArrayBuilderOptions& options, int64_t length);

    /// @brief Creates a GrowableBuffer from a full set of parameters.
    ///
    /// @param options Configuration options for building an array.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param length Currently used number of elements.
    /// @param reserved Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of reallocations.
    GrowableBuffer(const ArrayBuilderOptions& options,
                   std::shared_ptr<T> ptr,
                   int64_t length,
                   int64_t reserved);

    /// @brief Creates a GrowableBuffer by allocating a new buffer, taking an
    /// initial #reserved from
    /// {@link ArrayBuilderOptions#initial ArrayBuilderOptions::initial}.
    GrowableBuffer(const ArrayBuilderOptions& options);

    /// @brief Reference-counted pointer to the array buffer.
    const std::shared_ptr<T>
      ptr() const;

    /// @brief Currently used number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of reallocations.
    int64_t
      length() const;

    /// @brief Changes the #length in-place and possibly reallocate.
    ///
    /// If the `newlength` is larger than #reserved, #ptr is reallocated.
    void
      set_length(int64_t newlength);

    /// @brief Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of reallocations.
    int64_t
      reserved() const;

    /// @brief Possibly changes the #reserved and reallocate.
    ///
    /// The parameter only guarantees that at least `minreserved` is reserved;
    /// if an amount less than #reserved is requested, nothing changes.
    ///
    /// If #reserved actually changes, #ptr is reallocated.
    void
      set_reserved(int64_t minreserved);

    /// @brief Discards accumulated data, the #reserved returns to
    /// {@link ArrayBuilderOptions#initial ArrayBuilderOptions::initial},
    /// and a new #ptr is allocated.
    ///
    /// The old data are only discarded in the sense of decrementing their
    /// reference count. If any old snapshots are still using the data,
    /// they are not invalidated.
    void
      clear();

    /// @brief Inserts one `datum` into the array, possibly triggering a
    /// reallocation.
    ///
    /// This increases the #length by 1; if the new #length is larger than
    /// #reserved, a new #ptr will be allocated.
    void
      append(T datum);

    /// @brief Returns the element at a given position in the array, without
    /// handling negative indexing or bounds-checking.
    T
      getitem_at_nowrap(int64_t at) const;

  private:
    const ArrayBuilderOptions options_;
    // @brief See #ptr.
    std::shared_ptr<T> ptr_;
    // @brief See #length.
    int64_t length_;
    // @brief See #reserved.
    int64_t reserved_;
  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_

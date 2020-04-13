// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_PARTITIONEDARRAY_H_
#define AWKWARD_PARTITIONEDARRAY_H_

#include "awkward/Content.h"

namespace awkward {
  class PartitionedArray;
  using PartitionedArrayPtr = std::shared_ptr<PartitionedArray>;

  /// @class PartitionedArray
  ///
  /// @brief Abstract superclass of all PartitionedArray node types.
  /// PartitionedArrays contain a list of Content, but Content cannot contain
  /// PartitionedArrays.
  class EXPORT_SYMBOL PartitionedArray {
  public:
    PartitionedArray(const ContentPtrVec& partitions);

    /// @brief Empty destructor; required for some C++ reason.
    virtual ~PartitionedArray() { }

    /// @brief The partitions as a `std::vector<std::shared_ptr<Content>>`.
    const ContentPtrVec
      partitions() const;

    /// @brief The number of partitions in this array.
    int64_t
      numpartitions() const;

    /// @brief Returns a single partition as a `std::shared_ptr<Content>`.
    const ContentPtr
      partition(int64_t partitionid) const;

    /// @brief Get the partitionid and index for a given logical position in
    /// the full array, without handling negative indexing or bounds-checking.
    virtual void
      partitionid_index_at(int64_t at,
                           int64_t& partitionid,
                           int64_t& index) const override;

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief The length of the full array, summed over all partitions.
    virtual int64_t
      length() const = 0;

    /// @brief Copies this node without copying any nodes hierarchically
    /// nested within it or any array/index/identity buffers.
    ///
    /// See also #deep_copy.
    virtual const PartitionedArrayPtr
      shallow_copy() const = 0;

    /// @brief Returns the element at a given position in the array, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    const ContentPtr
      getitem_at(int64_t at) const;

    /// @brief Returns the element at a given position in the array, without
    /// handling negative indexing or bounds-checking.
    ///
    /// If the array has Identities, the identity bounds are checked.
    const ContentPtr
      getitem_at_nowrap(int64_t at) const;

    /// @brief Subinterval of this array, handling negative indexing
    /// and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    ///
    /// Ranges beyond the array are not an error; they are trimmed to
    /// `start = 0` on the left and `stop = length() - 1` on the right.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const;

    /// @brief Subinterval of this array, without handling negative
    /// indexing or bounds-checking.
    ///
    /// If the array has Identities, the identity bounds are checked.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    const ContentPtr
      getitem_range_nowrap(int64_t start, int64_t stop) const;

  private:
    const ContentPtrVec partitions_;
  };
}

#endif // AWKWARD_PARTITIONEDARRAY_H_

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

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
  class LIBAWKWARD_EXPORT_SYMBOL PartitionedArray {
  public:
    PartitionedArray(const ContentPtrVec& partitions);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~PartitionedArray();

    /// @brief The partitions as a `std::vector<std::shared_ptr<Content>>`.
    const ContentPtrVec
      partitions() const;

    /// @brief The number of partitions in this array.
    int64_t
      numpartitions() const;

    /// @brief Returns a single partition as a `std::shared_ptr<Content>`.
    const ContentPtr
      partition(int64_t partitionid) const;

    /// @brief Logical starting index for a given partitionid.
    virtual int64_t
      start(int64_t partitionid) const = 0;

    /// @brief Logical stopping index for a given partitionid.
    virtual int64_t
      stop(int64_t partitionid) const = 0;

    /// @brief Gets the partitionid and index for a given logical position in
    /// the full array, without handling negative indexing or bounds-checking.
    virtual void
      partitionid_index_at(int64_t at,
                           int64_t& partitionid,
                           int64_t& index) const = 0;

    /// @brief Returns this array with a specified (irregular) partitioning.
    virtual PartitionedArrayPtr
      repartition(const std::vector<int64_t>& stops) const = 0;

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Returns a string representation of this array (multi-line XML).
    ///
    /// Although this XML string has detail about every node in the tree,
    /// it does not show all elements in the array buffers and therefore
    /// does not scale with the size of the dataset (i.e. it is safe to
    /// print to the screen).
    ///
    /// Thus, it's also not a storage format: see #tojson.
    virtual const std::string
      tostring() const = 0;

    /// @brief Returns a JSON representation of this array.
    ///
    /// @param pretty If `true`, add spacing to make the JSON human-readable.
    /// If `false`, return a compact representation.
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for no limit.
    ///
    /// Although the JSON output contains all of the data from the array
    /// (and therefore might not be safe to print to the screen), it
    /// does not preserve the type information of an array. In particular,
    /// the distinction between ListType and RegularType is lost.
    const std::string
      tojson(bool pretty, int64_t maxdecimals) const;

    /// @brief Writes a JSON representation of this array to a `destination`
    /// file.
    ///
    /// @param destination The file to write into.
    /// @param pretty If `true`, add spacing to make the JSON human-readable.
    /// If `false`, return a compact representation.
    /// @param maxdecimals Maximum number of decimals for floating-point
    /// numbers or `-1` for no limit.
    /// @param buffersize Size of a temporary buffer in bytes.
    ///
    /// Although the JSON output contains all of the data from the array
    /// (and therefore might not be safe to print to the screen), it
    /// does not preserve the type information of an array. In particular,
    /// the distinction between ListType and RegularType is lost.
    void
      tojson(FILE* destination,
             bool pretty,
             int64_t maxdecimals,
             int64_t buffersize) const;

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
    const PartitionedArrayPtr
      getitem_range(int64_t start, int64_t stop, int64_t step) const;

    /// @brief Subinterval of this array, without handling negative
    /// indexing or bounds-checking.
    ///
    /// If the array has Identities, the identity bounds are checked.
    ///
    /// This operation only affects the node metadata; its calculation time
    /// does not scale with the size of the array.
    const PartitionedArrayPtr
      getitem_range_nowrap(int64_t start, int64_t stop, int64_t step) const;

    /// @brief Recursively copies components of the array from main memory to a
    /// GPU (if `ptr_lib == kernel::lib::cuda`) or to main memory (if
    /// `ptr_lib == kernel::lib::cpu`) if those components are not already there.
    virtual const PartitionedArrayPtr
      copy_to(kernel::lib ptr_lib) const = 0;

  protected:
    const ContentPtrVec partitions_;
  };
}

#endif // AWKWARD_PARTITIONEDARRAY_H_

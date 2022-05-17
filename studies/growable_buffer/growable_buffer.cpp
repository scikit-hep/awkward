#include <iostream>
#include <cmath>

namespace test {
 
struct GrowableBuffer {
    /// @brief Creates an empty GrowableBuffer.
    ///
    /// @param initial The initial number of reserved entries for a GrowableBuffer.
    static GrowableBuffer
      empty(const int initial) {
        return GrowableBuffer(0);
    }
 
    /// @brief Creates a GrowableBuffer by allocating a new buffer, taking an
    /// initial #reserved from initial

    GrowableBuffer(const int initial)
      : ptr_(new double[initial_])
      , length_(0)
      , reserved_(initial) { }
 
    /// @brief Reference to a pointer to the array buffer.
    const double*
      ptr() const { 
        return ptr_;
    }

    /// @brief Currently used number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of reallocations.
    int
      length() const {
        return length_; 
    }
 
    /// @brief Changes the #length in-place and possibly reallocate.
    ///
    /// If the `newlength` is larger than #reserved, #ptr is reallocated.
    void
      set_length(int newlength) { 
        if (newlength > reserved_) {
          set_reserved(newlength);
        }
        length_ = newlength;
    }
 
    /// @brief Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of reallocations.
    int
      reserved() const {
        return reserved_;
    }
 
    /// @brief Possibly changes the #reserved and reallocate.
    ///
    /// The parameter only guarantees that at least `minreserved` is reserved;
    /// if an amount less than #reserved is requested, nothing changes.
    ///
    /// If #reserved actually changes, #ptr is reallocated.
    void
      set_reserved(int minreserved) { 
        if (minreserved > reserved_) {
          double* ptr(new double[minreserved]);
          memcpy(ptr, ptr_, length_* sizeof(double));
          ptr_ = std::move(ptr);
          reserved_ = minreserved;
        }
   }
 
    /// @brief Discards accumulated data, the #reserved returns to
    /// {@link ArrayBuilderOptions#initial ArrayBuilderOptions::initial},
    /// and a new #ptr is allocated.
    void
      clear() { 
        length_ = 0;
        reserved_ = (int)initial_;
        ptr_ = new double[initial_];
    }
 
    /// @brief Inserts one `datum` into the array, possibly triggering a
    /// reallocation.
    ///
    /// This increases the #length by 1; if the new #length is larger than
    /// #reserved, a new #ptr will be allocated.
    void
      append(double datum) {
        if (length_ == reserved_) {
          set_reserved((int)ceil(reserved_ * 1.5));
        }
        ptr_[length_] = datum;
        length_++;
    }
 
    /// @brief Returns the element at a given position in the array, without
    /// handling negative indexing or bounds-checking.
    double
      getitem_at_nowrap(int at) const {
        return ptr_[at];
    }
    
    /// @brief Compacts all accumulated data from multiple panels to one 
    /// contiguously allocated memory panel 
    void snapshot() {  /* not implemented yet */  }
 
private:
    int initial_;
    double* ptr_;
    int length_;
    int reserved_;
 
};
 
}

 
int main(int argc, const char * argv[]) {
    int data_size = 13;
    double data[13] = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.1, 2.2, 2.3, 2.4};
    
    int initial = 10;
    auto buffer = test::GrowableBuffer::empty(initial);
    for (int i = 0; i < data_size; i++) {
        buffer.append(data[i]);
    }
    buffer.snapshot();
    for (int at = 0; at < buffer.length(); at++) {
        std::cout << buffer.getitem_at_nowrap(at) << ", ";
    }
    
    return 0;
}




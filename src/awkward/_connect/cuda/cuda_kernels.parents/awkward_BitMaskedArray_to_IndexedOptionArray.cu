// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_BitMaskedArray_to_IndexedOptionArray(
    T* toindex,
    const C* frombitmask,
    int64_t bitmasklength,
    bool validwhen,
    bool lsb_order,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (lsb_order) {
      if (thread_id < bitmasklength) {
        auto byte = frombitmask[thread_id];
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 0)] = ((thread_id * 8) + 0);

        } else {
          toindex[((thread_id * 8) + 0)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 1)] = ((thread_id * 8) + 1);

        } else {
          toindex[((thread_id * 8) + 1)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 2)] = ((thread_id * 8) + 2);

        } else {
          toindex[((thread_id * 8) + 2)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 3)] = ((thread_id * 8) + 3);

        } else {
          toindex[((thread_id * 8) + 3)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 4)] = ((thread_id * 8) + 4);

        } else {
          toindex[((thread_id * 8) + 4)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 5)] = ((thread_id * 8) + 5);

        } else {
          toindex[((thread_id * 8) + 5)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 6)] = ((thread_id * 8) + 6);

        } else {
          toindex[((thread_id * 8) + 6)] = -1;
        }
        byte >>= 1;
        if ((byte & (uint8_t)(1)) == validwhen) {
          toindex[((thread_id * 8) + 7)] = ((thread_id * 8) + 7);

        } else {
          toindex[((thread_id * 8) + 7)] = -1;
        }
      }

    } else {
      if (thread_id < bitmasklength) {
        auto byte = frombitmask[thread_id];
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 0)] = ((thread_id * 8) + 0);

        } else {
          toindex[((thread_id * 8) + 0)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 1)] = ((thread_id * 8) + 1);

        } else {
          toindex[((thread_id * 8) + 1)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 2)] = ((thread_id * 8) + 2);

        } else {
          toindex[((thread_id * 8) + 2)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 3)] = ((thread_id * 8) + 3);

        } else {
          toindex[((thread_id * 8) + 3)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 4)] = ((thread_id * 8) + 4);

        } else {
          toindex[((thread_id * 8) + 4)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 5)] = ((thread_id * 8) + 5);

        } else {
          toindex[((thread_id * 8) + 5)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 6)] = ((thread_id * 8) + 6);

        } else {
          toindex[((thread_id * 8) + 6)] = -1;
        }
        byte <<= 1;
        if (((byte & (uint8_t)(128)) != 0) == validwhen) {
          toindex[((thread_id * 8) + 7)] = ((thread_id * 8) + 7);

        } else {
          toindex[((thread_id * 8) + 7)] = -1;
        }
      }
    }
  }
}

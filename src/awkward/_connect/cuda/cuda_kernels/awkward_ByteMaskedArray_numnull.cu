// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_ByteMaskedArray_numnull(int64_t* numnull,
  				const int8_t* mask,
  				int64_t length,
  				bool validwhen,
                            	uint64_t invocation_index,
                            	uint64_t* err_code) {
  *numnull = 0;
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if ((mask[thread_id] != 0) != validwhen) {
      	*numnull = *numnull + 1;
      }
    }
  }
}

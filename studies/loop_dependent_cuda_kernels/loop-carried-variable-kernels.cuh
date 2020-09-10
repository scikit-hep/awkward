#include "standard_parallel_algorithms.cuh"

__global__
awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel(
	int* prefixed_mask,
	int* mask,
	int* to_carry,
	float* outindex,
	int validwhen) {

	int thread_id = threadIdx.x;

	if((mask[thread_id] != 0) == validwhen) {
		to_carry[prefixed_mask[thread_id] - 1] = thread_id;
		outindex[prefixed_mask[thread_id] - 1] = float(thread_id);
	} else {
		outindex[prefixed_mask[thread_id] - 1] = -1.0f;
	}
}

awkward_ByteMaskedArray_getitem_nextcarry_outindex(
	int** tocarry, 
	int** outindex, 
	int* mask, 
	int length, 
	int validwhen) {

	int* res_temp;

	HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int) * length));

	exclusive_scan(&res_temp, mask, length);

	awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel<<<1, length>>>(
		res_temp, 
		mask,
		*tocarry, 
		*outindex, 
		validwhen);

}


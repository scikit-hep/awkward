#include "standard_parallel_algorithms.cuh"
#include <iostream>

__global__
void awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel(
	int* prefixed_mask,
	int* mask,
	int* to_carry,
	float* outindex) {

	int thread_id = threadIdx.x;

	if(mask[thread_id] != 0) {
		to_carry[prefixed_mask[thread_id] - 1] = thread_id;
		outindex[prefixed_mask[thread_id] - 1] = float(thread_id);
	} else {
		outindex[prefixed_mask[thread_id] - 1] = -1.0f;
	}
}

__global__
void filter_mask(
	int* mask,
	bool validwhen) {
	int thread_id = threadIdx.x;

	if((mask[thread_id] != 0) == validwhen) {
		mask[thread_id] = 1;
	}

}


void awkward_ByteMaskedArray_getitem_nextcarry_outindex(
	int** tocarry, 
	float** outindex, 
	int* mask, 
	int length, 
	bool validwhen) {
	
	int* res_temp;
	int* h_mask = new int[length];


	HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int) * length));

	filter_mask<<<1, length>>>(
		mask,
		validwhen);		
	
	exclusive_scan(&res_temp, mask, length);
	
	awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel<<<1, length>>>(
		res_temp, 
		mask,
		*tocarry, 
		*outindex);


}


#include "standard_parallel_algorithms.cuh"
#include <iostream>

__global__
void awkward_Content_getitem_next_missing_jagged_getmaskstartstop_filter_mask(
	int* index_in,
	int* filtered_index) {
	int thread_id = threadIdx.x;

	if(index_in[thread_id] >= 0) {
		filtered_index[thread_id] = 1;
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
void awkward_ByteMaskedArray_getitem_nextcarry_kernel(
	int* prefixed_mask,
	int* mask,
	int* to_carry) {

	int thread_id = threadIdx.x;

	if(mask[thread_id] != 0) {
		to_carry[prefixed_mask[thread_id] - 1] = thread_id;
	}
}

__global__
void awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel(
	int* prefixed_index,
	int* index_in,
	int* offsets_in,
	int* mask_out,
	int* starts_out,
	int* stops_out) {

	int thread_id = threadIdx.x;
	int pre_in = prefixed_mask[thread_id] - 1;
	starts_out[thread_id] = offsets_in[pre_in];

	if(index_in[thread_id] < 0) {
		mask_out[thread_id] = -1;
		stops_out[thread_id] = offsets_in[pre_in];
	} else {
		mask_out[thread_id] = thread_id;
		stops_out[thread_id] = offsets_in[pre_in + 1];
	}
}

void awkward_ByteMaskedArray_getitem_nextcarry_outindex(
	int* tocarry, 
	float* outindex, 
	int* mask, 
	int length, 
	bool validwhen) {
	
	int* res_temp;
	int* filtered_mask;
	int* h_mask = new int[length];


	HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int) * length));
	HANDLE_ERROR(cudaMemcpy(filtered_mask, mask, sizeof(int) * length, cudaMemcpyDeviceToDevice));

	filter_mask<<<1, length>>>(
		filtered_mask,
		validwhen);		
	
	exclusive_scan(res_temp, filtered_mask, length);
	
	awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel<<<1, length>>>(
		res_temp, 
		filtered_mask,
		tocarry, 
		outindex);
}

void awkward_ByteMaskedArray_getitem_nextcarry(
	int* tocarry,
	int* mask,
	int length,
	bool validwhen) {

	int* res_temp;
	int* filtered_mask;
	int* h_mask = new int[length];


	HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int) * length));
	HANDLE_ERROR(cudaMemcpy(filtered_mask, mask, sizeof(int) * length, cudaMemcpyDeviceToDevice));

	filter_mask<<<1, length>>>(
		filtered_mask,
		validwhen);		
	
	exclusive_scan(res_temp, mask, length);
	
	awkward_ByteMaskedArray_getitem_nextcarry_kernel<<<1, length>>>(
		res_temp, 
		filtered_mask,
		tocarry);
}

void awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
	int* index_in,
	int* offsets_in,
	int* mask_out,
	int* starts_out,
	int* stops_out,
	int length) {

	int* res_temp;
	int* filtered_index;
	int* h_mask = new int[length];


	HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int) * length));
	HANDLE_ERROR(cudaMemcpy(filtered_index, mask, sizeof(int) * length, cudaMemcpyDeviceToDevice));

	awkward_Content_getitem_next_missing_jagged_getmaskstartstop_filter_mask<<<1, length>>>(
		index_in,
		filtered_index);		
	
	exclusive_scan(res_temp, filtered_index, length);

	awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel(
		index_in,
		offsets_in,
		mask_out,
		starts_out,
		stops_out);

}

	




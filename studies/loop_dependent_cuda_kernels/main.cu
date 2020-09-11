#include <debug/assertions.h>
#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include <stdexcept>
#include "loop-carried-variable-kernels.cuh"
#include <cassert>

void test_awkward_ByteMaskedArray_getitem_nextcarry_outindex() {
	// TEST 1
	int h_tocarry[] = {123, 123, 123};
	float h_outindex[] = {123, 123, 123};
	int h_mask[] = {1, 1, 1, 1, 1};
	int length = 3;
	bool valid_when = true;

	int* d_tocarry;
	float* d_outindex;
	int* d_mask;
	
	HANDLE_ERROR(cudaMalloc((void**)&d_tocarry, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&d_outindex, sizeof(float) * length));
	HANDLE_ERROR(cudaMalloc((void**)&d_mask, sizeof(int) * length));

	HANDLE_ERROR(cudaMemcpy(d_tocarry, h_tocarry, sizeof(h_tocarry), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_outindex, h_outindex, sizeof(h_outindex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_mask, h_mask, sizeof(int) * length, cudaMemcpyHostToDevice));

	awkward_ByteMaskedArray_getitem_nextcarry_outindex(
		d_tocarry, 
		d_outindex, 
		d_mask, 
		length, 
		valid_when);
	
	int* res_tocarry = new int[3];
	float* res_outindex = new float[3];

	int ground_tocarry[] = {0, 1, 2};
	float ground_outindex[] = {0.0, 1.0, 2.0};

	HANDLE_ERROR(cudaMemcpy(res_tocarry, d_tocarry, sizeof(int) * length, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(res_outindex, d_outindex, sizeof(float) * length, cudaMemcpyDeviceToHost));

	assert(!memcmp(res_tocarry, ground_tocarry, sizeof(int) * length) && !memcmp(res_outindex, ground_outindex, sizeof(float) * length));

	HANDLE_ERROR(cudaFree((void*)d_tocarry));
	HANDLE_ERROR(cudaFree((void*)d_outindex));
	HANDLE_ERROR(cudaFree((void*)d_mask));


	// TEST 2
	int h_mask_2[] = {0, 0, 0, 0, 0};
	length = 3;
	valid_when = false;

	HANDLE_ERROR(cudaMalloc((void**)&d_tocarry, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&d_outindex, sizeof(float) * length));
	HANDLE_ERROR(cudaMalloc((void**)&d_mask, sizeof(int) * length));

	HANDLE_ERROR(cudaMemcpy(d_tocarry, h_tocarry, sizeof(int) * length, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_outindex, h_outindex, sizeof(float) * length, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_mask, h_mask_2, sizeof(int) * length, cudaMemcpyHostToDevice));

	awkward_ByteMaskedArray_getitem_nextcarry_outindex(
		d_tocarry, 
		d_outindex, 
		d_mask, 
		length, 
		valid_when);

	int ground_tocarry_2[] = {0, 1, 2};
	float ground_outindex_2[] = {0.0, 1.0, 2.0};

	int* res_tocarry_2 = new int[3];
	float* res_outindex_2 = new float[3];

	HANDLE_ERROR(cudaMemcpy(res_tocarry_2, d_tocarry, sizeof(int) * length, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(res_outindex_2, d_outindex, sizeof(float) * length, cudaMemcpyDeviceToHost));
	
	assert(!memcmp(res_tocarry_2, ground_tocarry_2, sizeof(int) * length) && !memcmp(res_outindex_2, ground_outindex_2, sizeof(float) * length));
}

void test_awkward_ByteMaskedArray_getitem_nextcarry() {
	// TEST 1
	int h_tocarry[] = {123, 123, 123};
	int h_mask[] = {1, 1, 1, 1, 1};
	int length = 3;
	bool valid_when = true;

	int* d_tocarry;
	int* d_mask;
	
	HANDLE_ERROR(cudaMalloc((void**)&d_tocarry, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&d_mask, sizeof(int) * length));

	HANDLE_ERROR(cudaMemcpy(d_tocarry, h_tocarry, sizeof(h_tocarry), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_mask, h_mask, sizeof(int) * length, cudaMemcpyHostToDevice));

	awkward_ByteMaskedArray_getitem_nextcarry(
		d_tocarry, 
		d_mask, 
		length, 
		valid_when);
	
	int* res_tocarry = new int[3];
	float* res_outindex = new float[3];

	int ground_tocarry[] = {0, 1, 2};

	HANDLE_ERROR(cudaMemcpy(res_tocarry, d_tocarry, sizeof(int) * length, cudaMemcpyDeviceToHost));

	assert(!memcmp(res_tocarry, ground_tocarry, sizeof(int) * length));

	HANDLE_ERROR(cudaFree((void*)d_tocarry));
	HANDLE_ERROR(cudaFree((void*)d_mask));


	// TEST 2
	int h_mask_2[] = {0, 0, 0, 0, 0};
	length = 3;
	valid_when = false;

	HANDLE_ERROR(cudaMalloc((void**)&d_tocarry, sizeof(int) * length));
	HANDLE_ERROR(cudaMalloc((void**)&d_mask, sizeof(int) * length));

	HANDLE_ERROR(cudaMemcpy(d_tocarry, h_tocarry, sizeof(int) * length, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_mask, h_mask_2, sizeof(int) * length, cudaMemcpyHostToDevice));

	awkward_ByteMaskedArray_getitem_nextcarry(
		d_tocarry, 
		d_mask, 
		length, 
		valid_when);

	int ground_tocarry_2[] = {0, 1, 2};

	int* res_tocarry_2 = new int[3];
	float* res_outindex_2 = new float[3];

	HANDLE_ERROR(cudaMemcpy(res_tocarry_2, d_tocarry, sizeof(int) * length, cudaMemcpyDeviceToHost));
	
	assert(!memcmp(res_tocarry_2, ground_tocarry_2, sizeof(int) * length));
}

int main() {
	test_awkward_ByteMaskedArray_getitem_nextcarry_outindex();
}

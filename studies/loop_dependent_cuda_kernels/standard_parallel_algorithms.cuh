#include "cuErrorUtility.cuh"

__global__
void exclusive_scan_kernel(
	int* d_in,
	int* d_out,
	int* d_final,
	int curr_step,
	int total_steps,
	int stride,
	bool in_out_flag) {
	int thread_id = threadIdx.x;
	int sum = 0;	
	if(!in_out_flag) {
			if(thread_id < stride) {
					sum = d_out[thread_id];
					d_in[thread_id] = sum;
			} else {
					sum = d_out[thread_id] + d_out[thread_id - stride];
					d_in[thread_id] = sum;
			}
	} else {
			if(thread_id < stride) {
					sum = d_in[thread_id];
					d_out[thread_id] = sum;
			} else {
					sum = d_in[thread_id] + d_in[thread_id - stride];
					d_out[thread_id] = sum;
			}
	}

	if(curr_step == total_steps) {
			d_final[thread_id] = sum;
	}

}

__global__
void scatter(
	int* res_data_temp,
	int* d_temp,
	int* res_data_final) {
	int thread_id = threadIdx.x;
	if(d_temp[thread_id] == 1) {
			res_data_final[res_data_temp[thread_id] - 1] = thread_id;
	}
}

void exclusive_scan(
	int** out,
	int* in,
	int length) {

	int* d_in;
	int* d_out;
	int* d_final;

	HANDLE_ERROR(cudaMalloc((void**)&d_in, length * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_out, length * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_final, length * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(d_in, in, length * sizeof(int), cudaMemcpyDeviceToDevice));

	int stride = 1;
	int total_steps = ceil(log2(static_cast<float>(length)));

	for(int curr_step = 1; curr_step <= total_steps; curr_step++) {
		bool in_out_flag = (curr_step % 2) != 0;
		exclusive_scan_kernel<<<1, length>>>(
			d_in,
			d_out,
			d_final,
			curr_step,
			total_steps,
			stride,
			in_out_flag);
		stride = stride * 2;
	}

	(*out) = d_final;

}


#include <iostream>
#include <exception>
#include <sstream>

__global__
void exclusive_scan(int* d_in,
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
void scatter(int* res_data_temp,
		     int* d_temp,
		     int* res_data_final) {
		int thread_id = threadIdx.x;
		if(d_temp[thread_id] == 1) {
				res_data_final[res_data_temp[thread_id] - 1] = thread_id;
		}
}


int main() {
		int h_arr[] = {1, 1, 1, 0, 1, 0, 1};

		int* d_in; 
		int* d_out; 
		int* res_dest_final;
		int* res_dest_temp;
		int* d_temp;

		cudaError_t err;

		err = cudaMalloc((void**)&res_dest_final, sizeof(h_arr));

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out <<  "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}

		err = cudaMalloc((void**)&res_dest_temp, sizeof(h_arr));

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out <<  "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}

		err = cudaMalloc((void**)&d_temp, sizeof(h_arr));

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out <<  "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}

		err = cudaMalloc((void**)&d_in, sizeof(h_arr));

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out <<  "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}

		err = cudaMemcpy(d_in, h_arr, sizeof(h_arr), cudaMemcpyHostToDevice);

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out << "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}
		
		err = cudaMemcpy(d_temp, d_in, sizeof(h_arr), cudaMemcpyDeviceToDevice);

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out << "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}
		

		err = cudaMalloc((void**)&d_out, sizeof(h_arr));

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out <<  "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}


		int length = sizeof(h_arr) / sizeof(int);
		int stride = 1;
		int total_steps = ceil(log2(static_cast<float>(length)));

		for(int curr_step = 1; curr_step <= total_steps; curr_step++) {
				bool in_out_flag = (curr_step % 2) != 0;
				exclusive_scan<<<1, length>>>(d_in,
								              d_out,
										      res_dest_temp,
											  curr_step,
											  total_steps,
											  stride,
											  in_out_flag);
				stride = stride * 2;
		}

		scatter<<<1, length>>>(res_dest_temp, d_temp, res_dest_final);

		
		err = cudaMemcpy(h_arr, res_dest_final, sizeof(h_arr), cudaMemcpyDeviceToHost);

		if(err != cudaError_t::cudaSuccess) {
				std::stringstream out;
				out << "CUDA ERROR: " <<  cudaGetErrorString(err) << "\n";
				throw std::runtime_error(out.str());
		}

		for(int i = 0; i < sizeof(h_arr) / sizeof(int); i++) {
				std::cout << h_arr[i] << " ";
		}
		std::cout << std::endl;
}


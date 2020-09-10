#include <iostream>
#include <exception>
#include <sstream>
#include "standard_parallel_algorithms.cuh"

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


	exclusive_scan(&res_dest_temp, d_in, sizeof(h_arr) / sizeof(int));

	scatter<<<1, sizeof(h_arr) / sizeof(int)>>>(res_dest_temp, d_temp, res_dest_final);

	
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

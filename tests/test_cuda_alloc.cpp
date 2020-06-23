// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <memory>

#include <awkward/Index.h>
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/ArrayBuilder.h"
#include "awkward/kernel.h"
#include "awkward/cpu-kernels/operations.h"

#include "awkward/array/NumpyArray.h"

namespace ak = awkward;

int main(int, char**) {
//  std::shared_ptr<int64_t> main_index_arr(new int64_t [5], std::default_delete<int64_t []>());
//  main_index_arr.get()[0] = 0;
//  main_index_arr.get()[1] = 3;
//  main_index_arr.get()[2] = 4;
//  main_index_arr.get()[3] = 4;
//  main_index_arr.get()[4] = 5;
//
  std::shared_ptr<int32_t> main_offsets_arr(new int32_t[6],
    std::default_delete<int32_t[]>());
  main_offsets_arr.get()[0] = 0;
  main_offsets_arr.get()[1] = 3;
  main_offsets_arr.get()[2] = 3;
  main_offsets_arr.get()[3] = 5;
  main_offsets_arr.get()[4] = 6;
  main_offsets_arr.get()[5] = 10;

  ak::Index32 outoffsetsts(main_offsets_arr, 0, 6);
  ak::NumpyArray numpyArray(outoffsetsts);
  std::cout << numpyArray.format() << "\n";
  auto cuda_arr1 = numpyArray.to_gpu(kernel::Lib::cuda_kernels);
  std::cout << cuda_arr1->tostring() << "\n";
  auto cpu_arr = cuda_arr1->to_cpu();
  std::cout << cpu_arr->tostring() << "\n";
//  std::shared_ptr<ak::IndexOf<int8_t>> main_offsets = std::make_shared<ak::IndexOf<int8_t>>(main_offsets_arr, 0, 6);
//  auto cuda_arr = main_offsets->to_gpu(cuda_kernels);
//  std::cout << cuda_arr.tostring() << "\n";
//  std::cout << main_offsets->tostring() << "\n";
}




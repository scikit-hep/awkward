// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <memory>

#include <awkward/Index.h>

namespace ak = awkward;

int main(int, char**) {
  std::shared_ptr<int8_t> sh(new int8_t[10], std::default_delete<int8_t[]>());
  for(int i = 0;i < 10; i++) {
    sh.get()[i] = i + 1;
  }


  std::shared_ptr<ak::IndexOf<int8_t>> ptr_1 = std::make_shared<ak::IndexOf<int8_t>>(sh, 0, 10, cpu_kernels);
  auto cuda_ptr = ptr_1->to_cuda();
  std::cout << "Pointer " << (int64_t )cuda_ptr.getitem_at_nowrap(5) << "\n";
  std::shared_ptr<ak::IndexOf<uint8_t>> ptr_2 = std::make_shared<ak::IndexOf<uint8_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<int32_t>> ptr_3 = std::make_shared<ak::IndexOf<int32_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<uint32_t>> ptr_4 = std::make_shared<ak::IndexOf<uint32_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<int64_t>> ptr_5 = std::make_shared<ak::IndexOf<int64_t>>(8192, cuda_kernels);
  return 0;
}

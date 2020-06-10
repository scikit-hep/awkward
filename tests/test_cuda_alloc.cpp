// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <memory>

#include <awkward/Index.h>

namespace ak = awkward;

int main(int, char**) {
  std::shared_ptr<ak::IndexOf<int8_t>> ptr_1 = std::make_shared<ak::IndexOf<int8_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<uint8_t>> ptr_2 = std::make_shared<ak::IndexOf<uint8_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<int32_t>> ptr_3 = std::make_shared<ak::IndexOf<int32_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<uint32_t>> ptr_4 = std::make_shared<ak::IndexOf<uint32_t>>(8192, cuda_kernels);
  std::shared_ptr<ak::IndexOf<int64_t>> ptr_5 = std::make_shared<ak::IndexOf<int64_t>>(8192, cuda_kernels);
  return 0;
}

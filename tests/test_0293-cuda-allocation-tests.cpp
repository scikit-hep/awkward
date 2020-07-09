// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <memory>

#include <awkward/Index.h>
#include "awkward/kernel.h"
#include "awkward/array/NumpyArray.h"

namespace ak = awkward;

std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

// The extra steps needed to enable CUDA kernels on pure C++
class StartupLibraryPathCallback : public kernel::LibraryPathCallback {
public:
    StartupLibraryPathCallback() = default;

    const std::string library_path() const override {
      std::string cmd_string = exec("python -c 'import awkward1_cuda_kernels; print(awkward1_cuda_kernels.shared_library_path)'");

      std::string library_path = (cmd_string.substr(0, cmd_string.length() - 1));
      return library_path;
    };
};

int main(int, char**) {
  // Have access to cuda-kernels library on the C++ side
  kernel::lib_callback->add_library_path_callback(kernel::Lib::cuda_kernels,
                                                  std::make_shared<StartupLibraryPathCallback>());

  int8_t arr8[] = {1,2,3,4,5};
  uint8_t arrU8[] = {1,2,3,4,5};
  int32_t arr32[] = {1,2,3,4,5};
  uint32_t arrU32[] = {1,2,3,4,5};
  int64_t arr64[] = {1,2,3,4,5};
  float arrf[] = {1,2,3,4,5};

  // This is wrong practice, we can't have a stack allocated array wrapped with a
  // shared_ptr.
  std::shared_ptr<int8_t> main_arr8(arr8,
                                      kernel::array_deleter<int8_t>());
  std::shared_ptr<uint8_t> main_arrU8(arrU8,
                                      kernel::array_deleter<uint8_t>());
  std::shared_ptr<int32_t> main_arr32(arr32,
                                      kernel::array_deleter<int32_t>());
  std::shared_ptr<uint32_t> main_arrU32(arrU32,
                                        kernel::array_deleter<uint32_t>());
  std::shared_ptr<int64_t> main_arr64(arr64,
                                      kernel::array_deleter<int64_t>());
  std::shared_ptr<float> main_arrf(arrf,
                                    std::default_delete<float[]>());

  ak::Index8 index_arr8(main_arr8, 0, 5);
  ak::IndexU8 index_arrU8(main_arrU8, 0, 5);
  ak::Index32 index_arr32(main_arr32, 0, 5);
  ak::IndexU32 index_arrU32(main_arrU32, 0, 5);
  ak::Index64 index_arr64(main_arr64, 0, 5);

  auto cuda_arr8 = index_arr8.copy_to(kernel::Lib::cuda_kernels);
  std::cout << cuda_arr8.tostring() << "\n";
  auto cuda_arrU8 = index_arrU8.copy_to(kernel::Lib::cuda_kernels);
  std::cout << cuda_arrU8.tostring() << "\n";
  auto cuda_arr32 = index_arr32.copy_to(kernel::Lib::cuda_kernels);
  std::cout << cuda_arr32.tostring() << "\n";
  auto cuda_arrU32 = index_arrU32.copy_to(kernel::Lib::cuda_kernels);
  std::cout << cuda_arrU32.tostring() << "\n";
  auto cuda_arr64 = index_arr64.copy_to(kernel::Lib::cuda_kernels);
  std::cout << cuda_arr64.tostring() << "\n";

  std::cout << "\n";

  ak::NumpyArray numpyArray8(cuda_arr8);
  std::cout << numpyArray8.tostring() << "\n";
  ak::NumpyArray numpyArrayU8(cuda_arrU8);
  std::cout << numpyArrayU8.tostring() << "\n";
  ak::NumpyArray numpyArray32(cuda_arr32);
  std::cout << numpyArray32.tostring() << "\n";
  ak::NumpyArray numpyArrayU32(cuda_arrU32);
  std::cout << numpyArrayU32.tostring() << "\n";
  ak::NumpyArray numpyArray64(cuda_arr64);
  std::cout << numpyArray64.tostring() << "\n";

  std::cout << "\n";

  auto cpu_arr8 = numpyArray8.copy_to(kernel::Lib::cpu_kernels);
  std::cout << cpu_arr8->tostring() << "\n";
  auto cpu_arrU8 = numpyArrayU8.copy_to(kernel::Lib::cpu_kernels);
  std::cout << cpu_arrU8->tostring() << "\n";
  auto cpu_arr32 = numpyArray32.copy_to(kernel::Lib::cpu_kernels);
  std::cout << cpu_arr32->tostring() << "\n";
  auto cpu_arrU32 = numpyArrayU32.copy_to(kernel::Lib::cpu_kernels);
  std::cout << cpu_arrU32->tostring() << "\n";
  auto cpu_arr64 = numpyArray64.copy_to(kernel::Lib::cpu_kernels);
  std::cout << cpu_arr64->tostring() << "\n";

}




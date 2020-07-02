// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <memory>

#include "awkward/Slice.h"
#include "awkward/builder/ArrayBuilder.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/kernel.h"
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

void fill(ak::ArrayBuilder& builder, int64_t x) {
  builder.integer(x);
}

void fill(ak::ArrayBuilder& builder, double x) {
  builder.real(x);
}

void fill(ak::ArrayBuilder& builder, const char* x) {
  builder.bytestring(x);
}

void fill(ak::ArrayBuilder& builder, const std::string& x) {
  builder.bytestring(x.c_str());
}

template <typename T>
void fill(ak::ArrayBuilder& builder, const std::vector<T>& vector) {
  builder.beginlist();
  for (auto x : vector) {
    fill(builder, x);
  }
  builder.endlist();
}

int main(int, char**) {
  // Have access to cuda-kernels library on the C++ side
  kernel::lib_callback->add_library_path_callback(kernel::Lib::cuda_kernels,
                                                  std::make_shared<StartupLibraryPathCallback>());

  std::vector<std::vector<std::vector<double>>> vector =
    {{{0.0, 1.1, 2.2}, {}, {3.3, 4.4}}, {{5.5}}, {}, {{6.6, 7.7, 8.8, 9.9}}};

  ak::ArrayBuilder builder(ak::ArrayBuilderOptions(1024, 2.0));
  for (auto x : vector) fill(builder, x);
  std::shared_ptr<ak::Content> array = builder.snapshot();

  // First test, the transfer to GPU
  auto cuda_arr =  array->copy_to(kernel::Lib::cuda_kernels);
  std::cout << cuda_arr->tostring() << "\n";

  // Second test, run the ListArray_num kernel on the GPU
  auto arr_cuda_ker = cuda_arr->num(1, 0);
  std::cout << arr_cuda_ker->tostring() << "\n";

  // Third test, transfer the cuda array on to main memory
  auto cpu_arr = cuda_arr->copy_to(kernel::Lib::cpu_kernels);
  std::cout << cpu_arr->tostring() << "\n";

  // Fourth test, check the answer of ListArray num on the CPU and GPU
  auto arr_cpu_ker = array->num(1, 0);
  std::cout << arr_cpu_ker->tostring() << "\n";

  return 0;
}




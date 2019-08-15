#include <iostream>

#include "awkward/cpu-kernels/dummy1.h"

#ifdef VERSION_INFO
const char* version = VERSION_INFO;
#else
const char* version = "dev";
#endif

int dummy1(int x) {
  std::cout << version << std::endl;

  return x * x;
}

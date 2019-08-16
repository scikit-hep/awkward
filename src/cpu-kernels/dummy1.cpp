#include "awkward/cpu-kernels/dummy1.h"

__declspec(dllexport)
int dummy1(int x) {
  return x * x;
}

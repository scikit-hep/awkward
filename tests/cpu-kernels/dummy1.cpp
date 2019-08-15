#include <cassert>

#include "awkward/cpu-kernels/dummy1.h"

int main(int, char**) {
  assert(dummy1(3) == 9);
}

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/builder/ArrayBuilder.h"

#include <complex>
#include <cassert>

int main(int /* argc */, const char ** /* argv */) {

  auto a = awkward::ArrayBuilder(10);
  std::cout << "start length " << a.length() << std::endl;
  for (int64_t i = 0; i < 100; i++) {
    a.integer(1);
    std::cout << "length " << a.length() << std::endl;
    a.integer(2);
    std::cout << "length " << a.length() << std::endl;
  }
  assert(a.length() == 200);

  for (int64_t i = 0; i < 20; i++) {
    a.real(3.3);
    std::cout << "length " << a.length() << std::endl;
  }
  assert(a.length() == 220);

  for (int64_t i = 0; i < 2000; i++) {
    a.integer(4);
    std::cout << "length " << a.length() << std::endl;
    a.integer(5);
    std::cout << "length " << a.length() << std::endl;
  }
  assert(a.length() == 4220);

  std::cout << "final length " << a.length() << std::endl;

  return 0;
}

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include "awkward/BuilderOptions.h"
#include <cassert>

int main(int /* argc */, char ** /* argv */) {

  awkward::BuilderOptions options(1024, 1.0);

  assert(options.initial() == 1024);
  assert(options.resize() == 1.0);

  return 0;
}

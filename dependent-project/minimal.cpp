// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iostream>
#include <cstdlib>

#include "awkward/Content.h"
#include "awkward/io/json.h"

namespace ak = awkward;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "two arguments: JSON, int" << std::endl;
    return -1;
  }

  int at = std::atoi(argv[2]);

  std::shared_ptr<ak::Content> input = ak::FromJsonString(argv[1], ak::ArrayBuilderOptions(1024, 2.0));
  std::shared_ptr<ak::Content> selection = input.get()->getitem_at(at);

  std::cout << selection.get()->tojson(false, 1) << std::endl;
  
  return 0;
}

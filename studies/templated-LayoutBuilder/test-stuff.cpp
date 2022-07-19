#include <iostream>

#include "LayoutBuilder.h"

int main(int argc, char** argv) {
  std::cout << "BEGIN" << std::endl;

  ListOffsetLayoutBuilder<int64_t, NumpyLayoutBuilder<double>> builder;

  NumpyLayoutBuilder<double>* builder2 = builder.begin_list();
  builder2->append(1.1);
  builder2->append(2.2);
  builder2->append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  auto builder3 = builder.begin_list();
  builder3->append(4.4);
  builder3->append(5.5);
  builder.end_list();

  builder.dump("");

  std::cout << "END" << std::endl;

  return 0;
}

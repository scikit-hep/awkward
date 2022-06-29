// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "../src/awkward/_v2/cpp-headers/LayoutBuilder.h"

#include <cassert>

int main(int argc, char **argv) {

  static const unsigned initial = 10;

  auto builder = awkward::ListOffsetLayoutBuilder<
      initial, awkward::NumpyLayoutBuilder<initial, double>
  >();

  builder.begin_list();
  builder.append(1.1);
  builder.append(2.2);
  builder.append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder.append(4.4);
  builder.append(5.5);
  builder.end_list();

  builder.begin_list();
  builder.append(6.6);
  builder.end_list();

  builder.begin_list();
  builder.append(7.7);
  builder.append(8.8);
  builder.append(9.9);
  builder.end_list();

  auto form = builder.form(0);
  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}"
  );

  builder.dump(" ");

  return 0;
}

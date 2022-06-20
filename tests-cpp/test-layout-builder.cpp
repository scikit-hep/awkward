#include "../src/awkward/_v2/cpp-headers/LayoutBuilder.h"

#include <iostream>
#include <string>

void
test_list_of_numpy() {
  size_t initial = 10;
  awkward::ListOffsetLayoutBuilder<awkward::NumpyLayoutBuilder<double>> builder(initial);

  awkward::NumpyLayoutBuilder<double>* builder2 = builder.begin_list();
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
}

void
test_record_of_numpy() {
  awkward::RecordLayoutBuilder<awkward::NumpyLayoutBuilder<double>> builder;
  builder.begin_record();
  awkward::NumpyLayoutBuilder<double>* builder2 = builder.field(0);
  builder2->append(1);
  auto builder3 = builder.field(1);
  builder3->append(2);
  builder.end_record();

  builder.begin_record();
  auto builder4 = builder.field(0);
  builder4->append(1);
  auto builder5 = builder.field(1);
  builder5->append(2);
  builder.end_record();

  builder.begin_record();
  auto builder6 = builder.field(0);
  builder6->append(1);
  auto builder7 = builder.field(1);
  builder7->append(2);
  builder.end_record();
}

void
test_record_of_lists()
{
  awkward::RecordLayoutBuilder<awkward::ListOffsetLayoutBuilder<awkward::NumpyLayoutBuilder<double>>> builder;
  builder.begin_record();
  awkward::ListOffsetLayoutBuilder<awkward::NumpyLayoutBuilder<double>>* builder1 = builder.field(0);
  awkward::NumpyLayoutBuilder<double>* builder2 = builder1->begin_list();
  builder2->append(1.1);
  builder2->append(1.2);
  builder2->append(1.3);
  builder1->end_list();

  auto builder3 = builder.field(1);
  auto builder4 = builder3->begin_list();
  builder4->append(2.1);
  builder4->append(2.2);
  builder3->end_list();
  builder.end_record();

  builder.begin_record();
  awkward::ListOffsetLayoutBuilder<awkward::NumpyLayoutBuilder<double>>* builder5 = builder.field(0);
  awkward::NumpyLayoutBuilder<double>* builder6 = builder5->begin_list();
  builder6->append(3.1);
  builder6->append(3.2);
  builder6->append(3.3);
  builder5->end_list();

  auto builder7 = builder.field(1);
  auto builder8 = builder7->begin_list();
  builder8->append(4.1);
  builder8->append(4.2);
  builder7->end_list();
  builder.end_record();

  builder.dump("");
}

int main(int argc, char **argv) {
  std::cout << "BEGIN" << std::endl;
  size_t initial = 10;
  //test_list_of_numpy();
  //test_record_of_numpy();
  test_record_of_lists();
  std::cout << "END" << std::endl;
  return 0;
}

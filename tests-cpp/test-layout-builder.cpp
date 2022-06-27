#include "../src/awkward/_v2/cpp-headers/LayoutBuilder.h"

#include <iostream>
#include <cassert>
#include <complex>

void
test_record()
{
  static const char x_field[] = "one";
  static const char y_field[] = "two";
  static const char z_field[] = "three";

  static const unsigned initial = 10;

  auto builder = awkward::RecordLayoutBuilder<
        awkward::Record<awkward::field_name<x_field>, awkward::NumpyLayoutBuilder<initial, double>>,
        awkward::Record<awkward::field_name<y_field>, awkward::NumpyLayoutBuilder<initial, int64_t>>,
        awkward::Record<awkward::field_name<z_field>, awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>>
        >();

  auto form = builder.form();
  assert (form == "{ \"class\": \"RecordArray\", \"contents\": { \"one\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"two\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"int64\", \"form_key\": \"node2\" }, \"three\": "
                  "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node4\" }, \"form_key\": \"node3\" } }, "
                     "\"form_key\": \"node0\" }");

}

void
test_numpy() {
  static const unsigned initial = 10;
  auto builder = awkward::NumpyLayoutBuilder<initial, std::complex<double>>();

  builder.append({1.1, 0.1});
  builder.append({1.2, 0.2});
  builder.append({1.3, 0.3});
  builder.append({1.4, 0.4});
  builder.append({1.5, 0.5});

  auto form = builder.form();
  assert (form == "{ \"class\": \"NumpyArray\", \"primitive\": \"complex128\", \"form_key\": \"node0\" }");

  //builder.dump(" ");
}

void
test_listoffset_of_numpy() {
  static const unsigned initial = 10;
  auto builder = awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>();

  awkward::NumpyLayoutBuilder<initial, double>* builder2 = builder.begin_list();
  builder2->append(1.1);
  builder2->append(2.2);
  builder2->append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder2->append(4.4);
  builder2->append(5.5);
  builder.end_list();

  builder.begin_list();
  builder2->append(6.6);
  builder.end_list();

  builder.begin_list();
  builder2->append(7.7);
  builder2->append(8.8);
  builder2->append(9.9);
  builder.end_list();

  auto form = builder.form();
  assert (form == "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"form_key\": \"node0\" }");

  //builder.dump("");
}

void
test_listoffset_of_record() {
  static const unsigned initial = 10;
  static const char i_field[] = "i";
  static const char j_field[] = "j";

  auto builder = awkward::ListOffsetLayoutBuilder<initial, awkward::RecordLayoutBuilder<
                 awkward::Record<awkward::field_name<i_field>,awkward::NumpyLayoutBuilder<initial, int64_t>>,
                 awkward::Record<awkward::field_name<j_field>, awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>>
                 >>();

  auto form = builder.form();
  assert (form == "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
                  "{ \"class\": \"RecordArray\", \"contents\": { \"i\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"int64\", \"form_key\": \"node2\" }, \"j\": "
                  "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node4\" }, \"form_key\": \"node3\" } }, "
                  "\"form_key\": \"node1\" }, \"form_key\": \"node0\" }");

  //builder.dump("");
}

void
test_listoffset_of_listoffset() {
  static const unsigned initial = 10;
  auto builder = awkward::ListOffsetLayoutBuilder<initial, awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>>();

  awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>* builder2 = builder.begin_list();

  awkward::NumpyLayoutBuilder<initial, double>* builder3 = builder2->begin_list();
  builder3->append(1.1);
  builder3->append(2.2);
  builder3->append(3.3);
  builder2->end_list();

  builder2->begin_list();
  builder2->end_list();

  builder2->begin_list();
  builder3->append(4.4);
  builder3->append(5.5);
  builder2->end_list();

  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder2->begin_list();
  builder3->append(6.6);
  builder3->append(7.7);
  builder2->end_list();

  builder2->begin_list();
  builder3->append(8.8);
  builder3->append(9.9);
  builder2->end_list();
  builder.end_list();

  auto form = builder.form();
  assert (form == "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
                  "{ \"class\": \"ListOffsetArray\", \"offsets\": \"int64\", \"content\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node2\" }, "
                    "\"form_key\": \"node1\" }, \"form_key\": \"node0\" }");

  //builder.dump("");

}


int main(int argc, char **argv) {
  //test_record();
  //test_numpy();
  //test_listoffset_of_numpy();
  test_listoffset_of_record();
  //test_listoffset_of_listoffset();
  return 0;
}

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "../src/awkward/_v2/cpp-headers/LayoutBuilder.h"

#include <iostream>
#include <cassert>
#include <complex>

static const char one_field[] = "one";
static const char two_field[] = "two";
static const char three_field[] = "three";

static const char x_field[] = "x";
static const char y_field[] = "y";

static const char i_field[] = "i";
static const char j_field[] = "j";

static const char u_field[] = "u";
static const char v_field[] = "v";

static const unsigned initial = 10;

void
test_record()
{
  auto builder = awkward::RecordLayoutBuilder<0,
        awkward::Record<awkward::field_name<one_field>, awkward::NumpyLayoutBuilder<1, initial, double>>,
        awkward::Record<awkward::field_name<two_field>, awkward::NumpyLayoutBuilder<2, initial, double>>,
        awkward::Record<awkward::field_name<three_field>, awkward::ListOffsetLayoutBuilder<3, initial, awkward::NumpyLayoutBuilder<4, initial, double>>>
        >();

  auto x_builder = &(std::get<0>(builder.contents)->builder);
  auto y_builder = &(std::get<1>(builder.contents)->builder);
  auto z_builder = &(std::get<2>(builder.contents)->builder);

  builder.begin_record();

  x_builder->append(1.1);
  y_builder->append(2.1);

  z_builder->begin_list();
  z_builder->append(3.1);
  z_builder->append(3.2);
  z_builder->append(3.3);
  z_builder->end_list();

  builder.end_record();

  builder.clear();

  builder.begin_record();

  x_builder->append(1.2);
  y_builder->append(2.2);

  z_builder->begin_list();
  z_builder->end_list();

  builder.end_record();

  builder.begin_record();

  x_builder->append(1.3);
  y_builder->append(2.3);

  z_builder->begin_list();
  z_builder->append(3.4);
  z_builder->append(3.5);
  z_builder->end_list();

  builder.end_record();

  auto form = builder.form();
  std::cout << form << std::endl;
  assert (form == "{ \"class\": \"RecordArray\", \"contents\": { \"one\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"two\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node2\" }, \"three\": "
                  "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
                  "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node4\" }, \"form_key\": \"node3\" } }, "
                     "\"form_key\": \"node0\" }");

  builder.dump("");
}

void
test_nested_record()
{
  auto builder = awkward::RecordLayoutBuilder<0,
  awkward::Record<awkward::field_name<x_field>, awkward::ListOffsetLayoutBuilder<2, initial, awkward::RecordLayoutBuilder<3,
  awkward::Record<awkward::field_name<y_field>, awkward::NumpyLayoutBuilder<5, initial, double>>
  >>>>();

  auto form = builder.form();
  std::cout << form << std::endl;
}

void
test_nested_record_1()
{
  auto builder = awkward::RecordLayoutBuilder<0,
  awkward::Record<awkward::field_name<x_field>, awkward::ListOffsetLayoutBuilder<1, initial, awkward::NumpyLayoutBuilder<2, initial, double>>>,
  awkward::Record<awkward::field_name<y_field>, awkward::NumpyLayoutBuilder<3, initial, double>>>();

  int64_t form_key_id = 0;
  auto form = builder.form();
  std::cout << form << std::endl;
  assert (form == "{ \"class\": \"RecordArray\", \"contents\": { \"x\": "
    "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
    "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node2\" }, \"form_key\": \"node1\" }, \"y\": { \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node3\" } }, \"form_key\": \"node0\" }");
}
//
// void
// test_record_of_record()
// {
//   auto builder = awkward::RecordLayoutBuilder<
//   awkward::Record<awkward::field_name<u_field>, awkward::RecordLayoutBuilder<
//   awkward::Record<awkward::field_name<i_field>, awkward::NumpyLayoutBuilder<initial, double>>,
//   awkward::Record<awkward::field_name<j_field>, awkward::NumpyLayoutBuilder<initial, int>>
//   >>>();
//
//   auto form = builder.form();
//   std::cout << form << std::endl;
// }
//
// void
// test_numpy() {
//   static const unsigned initial = 10;
//   auto builder = awkward::NumpyLayoutBuilder<initial, std::complex<double>>();
//
//   builder.append({1.1, 0.1});
//   builder.append({1.2, 0.2});
//   builder.append({1.3, 0.3});
//   builder.append({1.4, 0.4});
//   builder.append({1.5, 0.5});
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"NumpyArray\", \"primitive\": \"complex128\", \"form_key\": \"node0\" }");
//
//   builder.dump(" ");
// }
//
// void
// test_listoffset_of_numpy() {
//   static const unsigned initial = 10;
//   auto builder = awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>();
//
//   awkward::NumpyLayoutBuilder<initial, double>* builder2 = builder.begin_list();
//   builder2->append(1.1);
//   builder2->append(2.2);
//   builder2->append(3.3);
//   builder.end_list();
//
//   builder.begin_list();
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->append(4.4);
//   builder2->append(5.5);
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->append(6.6);
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->append(7.7);
//   builder2->append(8.8);
//   builder2->append(9.9);
//   builder.end_list();
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   //builder.dump("");
// }
//
// void
// test_listoffset_of_record() {
//
//   auto builder = awkward::ListOffsetLayoutBuilder<initial, awkward::RecordLayoutBuilder<
//                  awkward::Record<awkward::field_name<i_field>,awkward::NumpyLayoutBuilder<initial, int64_t>>,
//                  awkward::Record<awkward::field_name<j_field>, awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>>
//                  >>();
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
//                   "{ \"class\": \"RecordArray\", \"contents\": { \"i\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"int64\", \"form_key\": \"node2\" }, \"j\": "
//                   "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node4\" }, \"form_key\": \"node3\" } }, "
//                   "\"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   //builder.dump("");
// }
//
// void
// test_listoffset_of_listoffset() {
//   auto builder = awkward::ListOffsetLayoutBuilder<initial, awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>>();
//
//   awkward::ListOffsetLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>* builder2 = builder.begin_list();
//
//   awkward::NumpyLayoutBuilder<initial, double>* builder3 = builder2->begin_list();
//   builder3->append(1.1);
//   builder3->append(2.2);
//   builder3->append(3.3);
//   builder2->end_list();
//
//   builder2->begin_list();
//   builder2->end_list();
//
//   builder2->begin_list();
//   builder3->append(4.4);
//   builder3->append(5.5);
//   builder2->end_list();
//
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->begin_list();
//   builder3->append(6.6);
//   builder2->end_list();
//   builder.end_list();
//
//   builder.begin_list();
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->begin_list();
//   builder3->append(7.7);
//   builder3->append(8.8);
//   builder2->end_list();
//
//   builder2->begin_list();
//   builder3->append(9.9);
//   builder2->end_list();
//   builder.end_list();
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
//                   "{ \"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node2\" }, "
//                     "\"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   builder.dump("");
//
// }
//
//
// void
// test_listarray_of_numpy() {
//   auto builder = awkward::ListLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>();
//
//   awkward::NumpyLayoutBuilder<initial, double>* builder2 = builder.begin_list();
//   builder2->append(1.1);
//   builder2->append(2.2);
//   builder2->append(3.3);
//   builder.end_list();
//
//   builder.begin_list();
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->append(4.4);
//   builder2->append(5.5);
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->append(6.6);
//   builder.end_list();
//
//   builder.begin_list();
//   builder2->append(7.7);
//   builder2->append(8.8);
//   builder2->append(9.9);
//   builder.end_list();
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"ListArray\", \"starts\": \"i64\", \"stops\": \"i64\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   builder.dump("");
// }
//
// void
// test_indexarray() {
//   auto builder = awkward::IndexedLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>();
//
//   builder.append(1.1);
//   builder.append(2.2);
//   builder.append(3.3);
//   builder.append(4.4);
//   builder.append(5.5);
//   builder.append(6.6);
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"IndexArray\", \"index\": \"i64\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   builder.dump("");
// }
//
// void
// test_indexoptionarray() {
//   auto builder = awkward::IndexedOptionLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>();
//
//   builder.append(1.1);
//   builder.append(2.2);
//   builder.null();
//   builder.append(3.3);
//   builder.append(4.4);
//   builder.append(5.5);
//   builder.null();
//
//   auto form = builder.form();
//   std::cout << form << std::endl;
//   assert (form == "{ \"class\": \"IndexedOptionArray\", \"index\": \"i64\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   builder.dump("");
// }
//
// void
// test_unmasked() {
//   auto builder = awkward::UnmaskedLayoutBuilder<initial, awkward::NumpyLayoutBuilder<initial, double>>();
//
//   builder.append(1.1);
//   builder.append(2.2);
//   builder.append(3.3);
//   builder.append(4.4);
//   builder.append(5.5);
//
//   auto form = builder.form();
//   assert (form == "{ \"class\": \"UnmaskedArray\", \"content\": "
//                   "{ \"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \"node1\" }, \"form_key\": \"node0\" }");
//
//   builder.dump("");
// }

int main(int argc, char **argv) {
  test_record();
  test_nested_record();
  test_nested_record_1();
  // test_record_of_record();
  // test_numpy();
  // test_listoffset_of_numpy();
  // test_listoffset_of_record();
  // test_listoffset_of_listoffset();
  // test_listarray_of_numpy();
  // test_indexarray();
  // test_indexoptionarray();
  // test_unmasked();
  return 0;
}

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "../src/awkward/_v2/cpp-headers/LayoutBuilder.h"

#include <iostream>

static const char one_field[] = "one";
static const char two_field[] = "two";
static const char three_field[] = "three";

static const char x_field[] = "x";
static const char y_field[] = "y";

static const char u_field[] = "u";
static const char v_field[] = "v";
static const char w_field[] = "w";

static const char i_field[] = "i";
static const char j_field[] = "j";

static const unsigned initial = 10;

template <class NODE, class PRIMITIVE, class LENGTH>
void dump(NODE&& node, PRIMITIVE&& ptr, LENGTH&& length) {
  std::cout << node << ": ";
  for (int at = 0; at < length; at++) {
    std::cout << ptr[at] << " ";
  }
  std::cout<<std::endl;
}

template<class NODE, class PRIMITIVE, class LENGTH, class ... Args>
void dump(NODE&& node, PRIMITIVE&& ptr, LENGTH&& length, Args&&...args)
{
    dump(node, ptr, length);
    dump(args...);
}

void
test_numpy_bool() {
  auto builder = awkward::NumpyLayoutBuilder<0, initial, bool>();

  builder.append(true);
  builder.append(false);
  builder.append(true);
  builder.append(true);

  // [True, False, True, True]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"bool\", "
      "\"form_key\": \"node0\" "
  "}");

  bool* ptr = new bool[builder.length()];
  builder.to_buffers(ptr);

  dump("node0", ptr, builder.length());
  std::cout<<std::endl;
}

void
test_numpy_int() {
  auto builder = awkward::NumpyLayoutBuilder<0, initial, int64_t>();

  size_t data_size = 10;

  int64_t data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

  builder.append(data, data_size);

 // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"int64\", "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.length()];
  builder.to_buffers(ptr0);

  dump("node0", ptr0, builder.length());
  std::cout<<std::endl;
}

void
test_numpy_char() {
  auto builder = awkward::NumpyLayoutBuilder<0, initial, char>();

  builder.append('a');
  builder.append('b');
  builder.append('c');
  builder.append('d');

  // ['a', 'b', 'c', 'd']

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"char\", "
      "\"form_key\": \"node0\" "
  "}");

  char* ptr0 = new char[builder.length()];
  builder.to_buffers(ptr0);

  dump("node0", ptr0, builder.length());
  std::cout<<std::endl;
}

void
test_numpy_double() {
  auto builder = awkward::NumpyLayoutBuilder<0, initial, double>();

  size_t data_size = 9;

  double data[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  builder.append(data, data_size);

  // [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"float64\", "
      "\"form_key\": \"node0\" "
  "}");

  double* ptr0 = new double[builder.length()];
  builder.to_buffers(ptr0);

  dump("node0", ptr0, builder.length());
  std::cout<<std::endl;
}

void
test_numpy_complex() {
  auto builder = awkward::NumpyLayoutBuilder<0, initial, std::complex<double>>();

  builder.append({1.1, 0.1});
  builder.append({2.2, 0.2});
  builder.append({3.3, 0.3});
  builder.append({4.4, 0.4});
  builder.append({5.5, 0.5});

  // [1.1 + 0.1j, 2.2 + 0.2j, 3.3 + 0.3j, 4.4 + 0.4j, 5.5 + 0.5j]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"complex128\", "
      "\"form_key\": \"node0\" "
  "}");

  std::complex<double>* ptr0 = new std::complex<double>[builder.length()];
  builder.to_buffers(ptr0);

  dump("node0", ptr0, builder.length());
  std::cout<<std::endl;
}

void
test_list_offset_of_numpy() {
  auto builder = awkward::ListOffsetLayoutBuilder<0,
      initial, awkward::NumpyLayoutBuilder<1, initial, double>
  >();

  auto builder2 = builder.begin_list();
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

  // [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

  auto form = builder.form();

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
  "}");

  int64_t* ptr0 = new int64_t[builder.length() + 1];
  builder.to_buffers(ptr0);

  double* ptr1 = new double[builder2->length()];
  builder2->to_buffers(ptr1);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, builder2->length());
  std::cout<<std::endl;
}

void
test_list_offset_of_list_offset() {
  auto builder = awkward::ListOffsetLayoutBuilder<0,
      initial, awkward::ListOffsetLayoutBuilder<1,
          initial, awkward::NumpyLayoutBuilder<2, initial, double>
  >>();

  auto builder2 = builder.begin_list();

  auto builder3 = builder2->begin_list();
  builder3->append(1.1);
  builder3->append(2.2);
  builder3->append(3.3);
  builder2->end_list();
  builder2->begin_list();
  builder2->end_list();
  builder.end_list();

  builder.begin_list();
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
  builder2->end_list();
  builder2->begin_list();
  builder3->append(7.7);
  builder3->append(8.8);
  builder3->append(9.9);
  builder2->end_list();
  builder.end_list();

  // [
  //     [[1.1, 2.2, 3.3], []],
  //     [[4.4, 5.5]],
  //     [],
  //     [[6.6], [7.7, 8.8, 9.9]],
  // ]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i64\", "
          "\"content\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node2\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.length() + 1];
  builder.to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[builder2->length() + 1];
  builder2->to_buffers(ptr1);

  double* ptr2 = new double[builder3->length()];
  builder3->to_buffers(ptr2);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, builder2->length() + 1,
       "node2", ptr2, builder3->length());
  std::cout<<std::endl;
}

void
test_record()
{
  auto builder = awkward::RecordLayoutBuilder<0,
      awkward::Record<awkward::field_name<one_field>, awkward::NumpyLayoutBuilder<1, initial, double>>,
      awkward::Record<awkward::field_name<two_field>, awkward::NumpyLayoutBuilder<2, initial, int64_t>>,
      awkward::Record<awkward::field_name<three_field>, awkward::NumpyLayoutBuilder<3, initial, char>>
  >();

  auto one_builder = &(std::get<0>(builder.contents)->builder);
  auto two_builder = &(std::get<1>(builder.contents)->builder);
  auto three_builder = &(std::get<2>(builder.contents)->builder);

  builder.begin_record();
  one_builder->append(1.1);
  two_builder->append(2);
  three_builder->append('a');
  builder.end_record();

  builder.begin_record();
  one_builder->append(3.3);
  two_builder->append(4);
  three_builder->append('b');
  builder.end_record();

  // [
  //     {"one": 1.1, "two": 2, "three": 'a'},
  //     {"one": 3.3, "two": 4. "three": 'b'},
  // ]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": { "
          "\"one\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node1\" "
          "}, "
          "\"two\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"int64\", "
              "\"form_key\": \"node2\" "
          "}, "
          "\"three\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"char\", "
              "\"form_key\": \"node3\" "
          "} "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  double* ptr0 = new double[one_builder->length()];
  one_builder->to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[two_builder->length()];
  two_builder->to_buffers(ptr1);

  char* ptr2 = new char[three_builder->length()];
  three_builder->to_buffers(ptr2);

  dump("node1", ptr0, one_builder->length(),
       "node2", ptr1, two_builder->length(),
       "node3", ptr2, three_builder->length());
  std::cout<<std::endl;
}

void
test_list_offset_of_record() {
  auto builder = awkward::ListOffsetLayoutBuilder<0,
      initial, awkward::RecordLayoutBuilder<1,
          awkward::Record<awkward::field_name<x_field>, awkward::NumpyLayoutBuilder<2, initial, double>>,
          awkward::Record<awkward::field_name<y_field>, awkward::ListOffsetLayoutBuilder<3,
              initial, awkward::NumpyLayoutBuilder<4, initial, int64_t>>
  >>>();

  auto builder2 = builder.begin_list();

  auto x_builder = &(std::get<0>(builder2->contents)->builder);
  auto y_builder = &(std::get<1>(builder2->contents)->builder);

  builder2->begin_record();
  x_builder->append(1.1);
  auto y_builder2 = y_builder->begin_list();
  y_builder->append(1);
  y_builder->end_list();
  builder2->end_record();

  builder2->begin_record();
  x_builder->append(2.2);
  y_builder->begin_list();
  y_builder->append(1);
  y_builder->append(2);
  y_builder->end_list();
  builder2->end_record();

  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();

  builder2->begin_record();
  x_builder->append(3.3);
  y_builder->begin_list();
  y_builder->append(1);
  y_builder->append(2);
  y_builder->append(3);
  y_builder->end_list();
  builder2->end_record();

  builder.end_list();

  // [
  //     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
  //     [],
  //     [{"x": 3.3, "y": [1, 2, 3]}],
  // ]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"RecordArray\", "
          "\"contents\": { "
              "\"x\": { "
                  "\"class\": \"NumpyArray\", "
                  "\"primitive\": \"float64\", "
                  "\"form_key\": \"node2\" "
              "}, "
              "\"y\": { "
                  "\"class\": \"ListOffsetArray\", "
                  "\"offsets\": \"i64\", "
                  "\"content\": { "
                      "\"class\": \"NumpyArray\", "
                      "\"primitive\": \"int64\", "
                      "\"form_key\": \"node4\" "
                  "}, "
                  "\"form_key\": \"node3\" "
              "} "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.length() + 1];
  builder.to_buffers(ptr0);

  double* ptr1 = new double[x_builder->length()];
  x_builder->to_buffers(ptr1);

  int64_t* ptr2 = new int64_t[y_builder->length() + 1];
  y_builder->to_buffers(ptr2);

  int64_t* ptr3 = new int64_t[y_builder2->length()];
  y_builder2->to_buffers(ptr3);

  dump("node0", ptr0, builder.length() + 1,
       "node2", ptr1, x_builder->length(),
       "node3", ptr2, y_builder->length() + 1,
       "node4", ptr3, y_builder2->length());
  std::cout<<std::endl;
}

void
test_record_of_record()
{
  auto builder = awkward::RecordLayoutBuilder<0,
      awkward::Record<awkward::field_name<x_field>, awkward::RecordLayoutBuilder<1,
          awkward::Record<awkward::field_name<u_field>, awkward::NumpyLayoutBuilder<2,initial, double>>,
          awkward::Record<awkward::field_name<v_field>, awkward::ListOffsetLayoutBuilder<3,
              initial, awkward::NumpyLayoutBuilder<4, initial, int64_t>>>>>,
      awkward::Record<awkward::field_name<y_field>, awkward::RecordLayoutBuilder<5,
          awkward::Record<awkward::field_name<w_field>, awkward::NumpyLayoutBuilder<6,initial, char>>>>
  >();

  auto x_builder = &(std::get<0>(builder.contents)->builder);
  auto y_builder = &(std::get<1>(builder.contents)->builder);

  auto u_builder = &(std::get<0>(x_builder->contents)->builder);
  auto v_builder = &(std::get<1>(x_builder->contents)->builder);

  auto w_builder = &(std::get<0>(y_builder->contents)->builder);

  builder.begin_record();

  x_builder->begin_record();
  u_builder->append(1.1);
  auto v_builder2 = v_builder->begin_list();
  v_builder->append(1);
  v_builder->append(2);
  v_builder->append(3);
  v_builder->end_list();
  x_builder->end_record();

  y_builder->begin_record();
  w_builder->append('a');
  y_builder->end_record();

  builder.end_record();

  builder.begin_record();

  x_builder->begin_record();
  u_builder->append(3.3);
  v_builder->begin_list();
  v_builder->append(4);
  v_builder->append(5);
  v_builder->end_list();
  x_builder->end_record();

  y_builder->begin_record();
  w_builder->append('b');
  y_builder->end_record();

  builder.end_record();

  // [
  //     {"x": {"u": 1.1, "v": [1, 2, 3]}, "y": {"w": 'a'}},
  //     {"x": {"u": 3.3, "v": [4, 5]}, "y": {"w": 'b'}},
  // ]

  auto form = builder.form();

  assert(form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": { "
          "\"x\": { "
              "\"class\": \"RecordArray\", "
              "\"contents\": { "
                  "\"u\": { "
                      "\"class\": \"NumpyArray\", "
                      "\"primitive\": \"float64\", "
                      "\"form_key\": \"node2\" "
                  "}, "
                  "\"v\": { "
                      "\"class\": \"ListOffsetArray\", "
                      "\"offsets\": \"i64\", "
                      "\"content\": { "
                          "\"class\": \"NumpyArray\", "
                          "\"primitive\": \"int64\", "
                          "\"form_key\": \"node4\" "
                      "}, "
                      "\"form_key\": \"node3\" "
                  "} "
              "}, "
              "\"form_key\": \"node1\" "
          "}, "
          "\"y\": { "
              "\"class\": \"RecordArray\", "
              "\"contents\": { "
                  "\"w\": { "
                      "\"class\": \"NumpyArray\", "
                      "\"primitive\": \"char\", "
                      "\"form_key\": \"node6\" "
                  "} "
              "}, "
              "\"form_key\": \"node5\" "
          "} "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  double* ptr0 = new double[u_builder->length()];
  u_builder->to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[v_builder->length() + 1];
  v_builder->to_buffers(ptr1);

  int64_t* ptr2 = new int64_t[v_builder2->length()];
  v_builder2->to_buffers(ptr2);

  char* ptr3 = new char[w_builder->length()];
  w_builder->to_buffers(ptr3);

  dump("node2", ptr0, u_builder->length(),
       "node3", ptr1, v_builder->length() + 1,
       "node4", ptr2, v_builder2->length(),
       "node6", ptr3, w_builder->length());
  std::cout << std::endl;
}

void
test_nested_record()
{
  auto builder = awkward::RecordLayoutBuilder<0,
      awkward::Record<awkward::field_name<u_field>, awkward::ListOffsetLayoutBuilder<1,
          initial, awkward::RecordLayoutBuilder<2,
              awkward::Record<awkward::field_name<i_field>, awkward::NumpyLayoutBuilder<3, initial, double>>,
              awkward::Record<awkward::field_name<j_field>, awkward::ListOffsetLayoutBuilder<4,
                  initial, awkward::NumpyLayoutBuilder<5, initial, int64_t>>>>>>,
      awkward::Record<awkward::field_name<v_field>, awkward::NumpyLayoutBuilder<6, initial, int64_t>>,
      awkward::Record<awkward::field_name<w_field>, awkward::NumpyLayoutBuilder<7, initial, double>>
  >();

  auto u_builder = &(std::get<0>(builder.contents)->builder);
  auto v_builder = &(std::get<1>(builder.contents)->builder);
  auto w_builder = &(std::get<2>(builder.contents)->builder);

  builder.begin_record();

  auto u_builder2 = u_builder->begin_list();

  auto i_builder = &(std::get<0>(u_builder2->contents)->builder);
  auto j_builder = &(std::get<1>(u_builder2->contents)->builder);

  u_builder2->begin_record();
  i_builder->append(1.1);
  auto j_builder2 = j_builder->begin_list();
  j_builder->append(1);
  j_builder->append(2);
  j_builder->append(3);
  j_builder->end_list();
  u_builder2->end_record();

  u_builder->end_list();

  v_builder->append(-1);
  w_builder->append(3.3);

  builder.end_record();

  builder.begin_record();

  u_builder->begin_list();

  u_builder2->begin_record();
  i_builder->append(2.2);
  j_builder->begin_list();
  j_builder->append(4);
  j_builder->append(5);
  j_builder->end_list();
  u_builder2->end_record();

  u_builder->end_list();

  v_builder->append(-2);
  w_builder->append(4.4);

  builder.end_record();

  // [
  //     {"u": [{"i": 1.1, "j": [1, 2, 3]}], "v": -1, "w": 3.3},
  //     {"u": [{"i": 2.2, "j": [4, 5]}], "v": -2, "w": 4.4},
  // ]

  auto form = builder.form();
  assert (form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": { "
          "\"u\": { "
              "\"class\": \"ListOffsetArray\", "
              "\"offsets\": \"i64\", "
              "\"content\": { "
                  "\"class\": \"RecordArray\", "
                  "\"contents\": { "
                      "\"i\": { "
                          "\"class\": \"NumpyArray\", "
                          "\"primitive\": \"float64\", "
                          "\"form_key\": \"node3\" "
                      "}, "
                      "\"j\": { "
                          "\"class\": \"ListOffsetArray\", "
                          "\"offsets\": \"i64\", "
                          "\"content\": { "
                              "\"class\": \"NumpyArray\", "
                              "\"primitive\": \"int64\", "
                              "\"form_key\": \"node5\" "
                          "}, "
                          "\"form_key\": \"node4\" "
                      "} "
                  "}, "
                  "\"form_key\": \"node2\" "
              "}, "
              "\"form_key\": \"node1\" "
          "}, "
          "\"v\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"int64\", "
              "\"form_key\": \"node6\" "
          "}, "
          "\"w\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node7\" "
          "} "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[u_builder->length() + 1];
  u_builder->to_buffers(ptr0);

  double* ptr1 = new double[i_builder->length()];
  i_builder->to_buffers(ptr1);

  int64_t* ptr2 = new int64_t[j_builder->length() + 1];
  j_builder->to_buffers(ptr2);

  int64_t* ptr3 = new int64_t[j_builder2->length()];
  j_builder2->to_buffers(ptr3);

  int64_t* ptr4 = new int64_t[v_builder->length()];
  v_builder->to_buffers(ptr4);

  double* ptr5 = new double[w_builder->length()];
  w_builder->to_buffers(ptr5);

  dump("node1", ptr0, u_builder->length() + 1,
       "node3", ptr1, i_builder->length(),
       "node4", ptr2, j_builder->length() + 1,
       "node5", ptr3, j_builder2->length(),
       "node6", ptr4, v_builder->length(),
       "node7", ptr5, w_builder->length());
  std::cout << std::endl;
}

void
test_list() {
  auto builder = awkward::ListLayoutBuilder<0, initial, awkward::NumpyLayoutBuilder<1, initial, double>>();

  auto builder2 = builder.begin_list();
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

  // [
  //     [1.1, 2.2, 3.3],
  //     [],
  //     [4.4, 5.5],
  //     [6.6],
  //     [7.7, 8.8, 9.9],
  // ]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListArray\", "
      "\"starts\": \"i64\", "
      "\"stops\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr1 = new int64_t[builder.length()];
  int64_t* ptr2 = new int64_t[builder.length()];
  builder.to_buffers(ptr1, ptr2);
  double* ptr3 = new double[builder2->length()];
  builder2->to_buffers(ptr3);
  dump("node0", ptr1, builder.length(),
       "     ", ptr2, builder.length(),
       "node1", ptr3, builder2->length());
  std::cout<<std::endl;
}

void
test_index() {
  auto builder = awkward::IndexedLayoutBuilder<0, initial, awkward::NumpyLayoutBuilder<1, initial, double>>();

  size_t data_size = 9;
  double data[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  builder.append(data, data_size);

  auto form = builder.form();

  // [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

  assert (form ==
  "{ "
      "\"class\": \"IndexedArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.length()];
  builder.to_buffers(ptr0);

  double* ptr1 = new double[builder.content()->length()];
  builder.content()->to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content()->length());
  std::cout<<std::endl;
}

void
test_index_option() {
  auto builder = awkward::IndexedOptionLayoutBuilder<0, initial, awkward::NumpyLayoutBuilder<1, initial, int64_t>>();

  builder.append(11);
  builder.append(22);
  builder.null();
  builder.append(33);
  builder.append(44);
  builder.append(55);
  builder.null();
  builder.null();
  builder.append(66);
  builder.append(77);

  // [11, 22, None, 33, 44, 55, None, None, 66, 77]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"int64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.length()];
  builder.to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[builder.content()->length()];
  builder.content()->to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content()->length());
  std::cout<<std::endl;
}

void
test_empty() {
  auto builder = awkward::EmptyLayoutBuilder<1>();

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"EmptyArray\" "
  "}");
}

void
test_list_offset_of_empty() {
  auto builder = awkward::ListOffsetLayoutBuilder<0,
      initial, awkward::ListOffsetLayoutBuilder<1,
          initial, awkward::EmptyLayoutBuilder<2>
  >>();

  builder.begin_list();
  builder.end_list();

  auto builder2 = builder.begin_list();
  builder2->begin_list();
  builder2->end_list();
  builder2->begin_list();
  builder2->end_list();
  builder2->begin_list();
  builder2->end_list();
  builder.end_list();

  builder.begin_list();
  builder2->begin_list();
  builder2->end_list();
  builder2->begin_list();
  builder2->end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder2->begin_list();
  builder2->end_list();
  builder.end_list();

  //  [[], [[], [], []], [[], []], [], [[]]]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i64\", "
          "\"content\": { "
              "\"class\": \"EmptyArray\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.length() + 1];
  builder.to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[builder2->length() + 1];
  builder2->to_buffers(ptr1);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, builder2->length() + 1);
  std::cout<<std::endl;
}

void
test_unmasked() {
  auto builder = awkward::UnmaskedLayoutBuilder<9, awkward::NumpyLayoutBuilder<10, initial, int64_t>>();

  builder.append(-1);
  builder.append(-2);
  builder.append(-3);
  builder.append(-4);
  builder.append(-5);

  // [-1, -2, -3, -4, -5]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"UnmaskedArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"int64\", "
          "\"form_key\": \"node10\" "
      "}, "
      "\"form_key\": \"node9\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.content()->length()];
  builder.content()->to_buffers(ptr0);

  dump("node0", ptr0, builder.content()->length());
  std::cout<<std::endl;
}

int main(int argc, char **argv) {
  test_numpy_bool();
  test_numpy_char();
  test_numpy_int();
  test_numpy_double();
  test_numpy_complex();
  test_list_offset_of_numpy();
  test_list_offset_of_list_offset();
  test_record();
  test_list_offset_of_record();
  test_record_of_record();
  test_nested_record();
  test_list();
  test_index();
  test_index_option();
  test_unmasked();
  test_empty();
  test_list_offset_of_empty();

  return 0;
}

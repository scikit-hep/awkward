// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "../src/awkward/_v2/cpp-headers/LayoutBuilder.h"

namespace lb = awkward::LayoutBuilder;

static const char param[] = "";

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
  for (size_t at = 0; at < length; at++) {
    std::cout << +ptr[at] << " ";
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
test_Numpy_bool() {
  auto builder = lb::Numpy<initial, bool>();

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
test_Numpy_int() {
  auto builder = lb::Numpy<initial, int64_t>();

  size_t data_size = 10;

  int64_t data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

  builder.extend(data, data_size);

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
test_Numpy_char() {
  auto builder = lb::Numpy<initial, char>();

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
test_Numpy_double() {
  auto builder = lb::Numpy<initial, double>();

  size_t data_size = 9;

  double data[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  builder.extend(data, data_size);

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
test_Numpy_complex() {
  auto builder = lb::Numpy<initial, std::complex<double>>();

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
test_ListOffset() {
  auto builder = lb::ListOffset<initial, lb::Numpy<initial, double>>();

  auto subbuilder = builder.begin_list();
  subbuilder->append(1.1);
  subbuilder->append(2.2);
  subbuilder->append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder->append(4.4);
  subbuilder->append(5.5);
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

  double* ptr1 = new double[subbuilder->length()];
  subbuilder->to_buffers(ptr1);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, subbuilder->length());
  std::cout<<std::endl;
}

void
test_ListOffset_ListOffset() {
  auto builder = lb::ListOffset<initial,
      lb::ListOffset<initial,
          lb::Numpy<initial, double>
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
test_EmptyRecord() {
  auto builder = lb::EmptyRecord<true>();

  builder.append();

  builder.extend(2);

  // [(), (), ()]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": [], "
      "\"form_key\": \"node0\" "
  "}");
}

void
test_Record()
{
  auto builder = lb::Record<
      lb::Field<lb::field_name<one_field>, lb::Numpy<initial, double>>,
      lb::Field<lb::field_name<two_field>, lb::Numpy<initial, int64_t>>,
      lb::Field<lb::field_name<three_field>, lb::Numpy<initial, char>>
  >();

  auto one_builder = &(std::get<0>(builder.contents)->builder);
  auto two_builder = &(std::get<1>(builder.contents)->builder);
  auto three_builder = &(std::get<2>(builder.contents)->builder);

  one_builder->append(1.1);
  two_builder->append(2);
  three_builder->append('a');

  one_builder->append(3.3);
  two_builder->append(4);
  three_builder->append('b');

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
test_ListOffset_Record() {
  auto builder = lb::ListOffset<initial,
      lb::Record<
          lb::Field<lb::field_name<x_field>, lb::Numpy<initial, double>>,
          lb::Field<lb::field_name<y_field>, lb::ListOffset<initial,
              lb::Numpy<initial, int32_t>>
  >>>();

  auto subbuilder = builder.begin_list();

  auto x_builder = &(std::get<0>(subbuilder->contents)->builder);
  auto y_builder = &(std::get<1>(subbuilder->contents)->builder);

  x_builder->append(1.1);
  auto y_subbuilder = y_builder->begin_list();
  y_subbuilder->append(1);
  y_builder->end_list();

  x_builder->append(2.2);
  y_builder->begin_list();
  y_subbuilder->append(1);
  y_subbuilder->append(2);
  y_builder->end_list();

  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();

  x_builder->append(3.3);
  y_builder->begin_list();
  y_subbuilder->append(1);
  y_subbuilder->append(2);
  y_subbuilder->append(3);
  y_builder->end_list();

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
                      "\"primitive\": \"int32\", "
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

  int32_t* ptr3 = new int32_t[y_subbuilder->length()];
  y_subbuilder->to_buffers(ptr3);

  dump("node0", ptr0, builder.length() + 1,
       "node2", ptr1, x_builder->length(),
       "node3", ptr2, y_builder->length() + 1,
       "node4", ptr3, y_subbuilder->length());
  std::cout<<std::endl;
}

void
test_Record_Record()
{
  auto builder = lb::Record<
      lb::Field<lb::field_name<x_field>, lb::Record<
          lb::Field<lb::field_name<u_field>, lb::Numpy<initial, double>>,
          lb::Field<lb::field_name<v_field>, lb::ListOffset<initial,
              lb::Numpy<initial, int64_t>>>>>,
      lb::Field<lb::field_name<y_field>, lb::Record<
          lb::Field<lb::field_name<w_field>, lb::Numpy<initial, char>>>>
  >();

  auto x_builder = &(std::get<0>(builder.contents)->builder);
  auto y_builder = &(std::get<1>(builder.contents)->builder);

  auto u_builder = &(std::get<0>(x_builder->contents)->builder);
  auto v_builder = &(std::get<1>(x_builder->contents)->builder);

  auto w_builder = &(std::get<0>(y_builder->contents)->builder);


  u_builder->append(1.1);
  auto v_subbuilder = v_builder->begin_list();
  v_subbuilder->append(1);
  v_subbuilder->append(2);
  v_subbuilder->append(3);
  v_builder->end_list();

  w_builder->append('a');

  u_builder->append(3.3);
  v_builder->begin_list();
  v_subbuilder->append(4);
  v_subbuilder->append(5);
  v_builder->end_list();

  w_builder->append('b');

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

  int64_t* ptr2 = new int64_t[v_subbuilder->length()];
  v_subbuilder->to_buffers(ptr2);

  char* ptr3 = new char[w_builder->length()];
  w_builder->to_buffers(ptr3);

  dump("node2", ptr0, u_builder->length(),
       "node3", ptr1, v_builder->length() + 1,
       "node4", ptr2, v_subbuilder->length(),
       "node6", ptr3, w_builder->length());
  std::cout << std::endl;
}

void
test_Record_nested()
{
  auto builder = lb::Record<
      lb::Field<lb::field_name<u_field>, lb::ListOffset<initial,
          lb::Record<
              lb::Field<lb::field_name<i_field>, lb::Numpy<initial, double>>,
              lb::Field<lb::field_name<j_field>, lb::ListOffset<initial,
                  lb::Numpy<initial, int64_t>>>>>>,
      lb::Field<lb::field_name<v_field>, lb::Numpy<initial, int64_t>>,
      lb::Field<lb::field_name<w_field>, lb::Numpy<initial, double>>
  >();

  auto u_builder = &(std::get<0>(builder.contents)->builder);
  auto v_builder = &(std::get<1>(builder.contents)->builder);
  auto w_builder = &(std::get<2>(builder.contents)->builder);

  auto u_subbuilder = u_builder->begin_list();

  auto i_builder = &(std::get<0>(u_subbuilder->contents)->builder);
  auto j_builder = &(std::get<1>(u_subbuilder->contents)->builder);

  i_builder->append(1.1);
  auto j_subbuilder = j_builder->begin_list();
  j_subbuilder->append(1);
  j_subbuilder->append(2);
  j_subbuilder->append(3);
  j_builder->end_list();

  u_builder->end_list();

  v_builder->append(-1);
  w_builder->append(3.3);

  u_builder->begin_list();

  i_builder->append(2.2);
  j_builder->begin_list();
  j_subbuilder->append(4);
  j_subbuilder->append(5);
  j_builder->end_list();

  u_builder->end_list();

  v_builder->append(-2);
  w_builder->append(4.4);

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

  int64_t* ptr3 = new int64_t[j_subbuilder->length()];
  j_subbuilder->to_buffers(ptr3);

  int64_t* ptr4 = new int64_t[v_builder->length()];
  v_builder->to_buffers(ptr4);

  double* ptr5 = new double[w_builder->length()];
  w_builder->to_buffers(ptr5);

  dump("node1", ptr0, u_builder->length() + 1,
       "node3", ptr1, i_builder->length(),
       "node4", ptr2, j_builder->length() + 1,
       "node5", ptr3, j_subbuilder->length(),
       "node6", ptr4, v_builder->length(),
       "node7", ptr5, w_builder->length());
  std::cout << std::endl;
}

void
test_List() {
  auto builder = lb::List<initial, lb::Numpy<initial, double>>();

  auto subbuilder = builder.begin_list();
  subbuilder->append(1.1);
  subbuilder->append(2.2);
  subbuilder->append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder->append(4.4);
  subbuilder->append(5.5);
  builder.end_list();

  builder.begin_list();
  subbuilder->append(6.6);

  builder.end_list();

  builder.begin_list();
  subbuilder->append(7.7);
  subbuilder->append(8.8);
  subbuilder->append(9.9);
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
  double* ptr3 = new double[subbuilder->length()];
  subbuilder->to_buffers(ptr3);
  dump("node0", ptr1, builder.length(),
       "     ", ptr2, builder.length(),
       "node1", ptr3, subbuilder->length());
  std::cout<<std::endl;
}

void
test_Indexed() {
  auto builder = lb::Indexed<initial, lb::Numpy<initial, double>>();

  auto subbuilder = builder.append_index();
  subbuilder->append(1.1);

  builder.append_index();
  subbuilder->append(2.2);

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder->extend(data, 3);

  // [1.1, 2.2, 3.3, 4.4, 5.5]

  auto form = builder.form();

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
test_IndexedOption() {
  auto builder = lb::IndexedOption<initial, lb::Numpy<initial, double>>();

  auto subbuilder = builder.append_index();
  subbuilder->append(1.1);

  builder.append_null();

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder->extend(data, 3);

  builder.extend_null(2);

  // [1.1, None, 3.3, 4.4, 5.5, None, None]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
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
test_Empty() {
  auto builder = lb::Empty();

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"EmptyArray\" "
  "}");
}

void
test_ListOffset_Empty() {
  auto builder = lb::ListOffset<initial,
      lb::ListOffset<initial, lb::Empty
  >>();

  builder.begin_list();
  builder.end_list();

  auto subbuilder = builder.begin_list();
  subbuilder->begin_list();
  subbuilder->end_list();
  subbuilder->begin_list();
  subbuilder->end_list();
  subbuilder->begin_list();
  subbuilder->end_list();
  builder.end_list();

  builder.begin_list();
  subbuilder->begin_list();
  subbuilder->end_list();
  subbuilder->begin_list();
  subbuilder->end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder->begin_list();
  subbuilder->end_list();
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

  int64_t* ptr1 = new int64_t[subbuilder->length() + 1];
  subbuilder->to_buffers(ptr1);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, subbuilder->length() + 1);
  std::cout<<std::endl;
}

void
test_Unmasked() {
  auto builder = lb::Unmasked<lb::Numpy<initial, int64_t>>();

  auto subbuilder = builder.append_valid();
  subbuilder->append(11);
  subbuilder->append(22);
  subbuilder->append(33);
  subbuilder->append(44);
  subbuilder->append(55);

  // [11, 22, 33, 44, 55]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"UnmaskedArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"int64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  int64_t* ptr0 = new int64_t[builder.content()->length()];
  builder.content()->to_buffers(ptr0);

  dump("node0", ptr0, builder.content()->length());
  std::cout<<std::endl;
}

void
test_ByteMasked() {
  auto builder = lb::ByteMasked<true, initial,
      lb::Numpy<initial, double>
  >();

  auto subbuilder = builder.append_valid();
  subbuilder->append(1.1);

  builder.append_null();
  subbuilder->append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder->extend(data, 3);

  builder.extend_null(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder->append(-1000);  // have to supply a "dummy" value
  }

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"ByteMaskedArray\", "
      "\"mask\": \"i8\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"valid_when\": true, "
      "\"form_key\": \"node0\" "
  "}");

  int8_t* ptr0 = new int8_t[builder.length()];
  builder.to_buffers(ptr0);

  double* ptr1 = new double[builder.content()->length()];
  builder.content()->to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content()->length());
  std::cout<<std::endl;
}

void
test_BitMasked() {
  auto builder = lb::BitMasked<true, true, initial,
      lb::Numpy<initial, double>
  >();

  auto subbuilder = builder.append_valid();
  subbuilder->append(1.1);

  builder.append_null();
  subbuilder->append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder->extend(data, 3);

  builder.extend_null(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder->append(-1000);  // have to supply a "dummy" value
  }

  builder.append_valid();
  subbuilder->append(8);

  builder.append_valid();
  subbuilder->append(9);

  builder.append_valid();
  subbuilder->append(10);

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000, 8, 9, 10]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"BitMaskedArray\", "
      "\"mask\": \"u8\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"valid_when\": true, "
      "\"lsb_order\": true, "
      "\"form_key\": \"node0\" "
  "}");

  uint8_t* ptr0 = new uint8_t[builder.length()];
  builder.to_buffers(ptr0);

  double* ptr1 = new double[builder.content()->length()];
  builder.content()->to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content()->length());
  std::cout<<std::endl;
}

void
test_Regular() {
  auto builder = lb::Regular<3,
      lb::Numpy<initial, double>
  >();

  auto subbuilder = builder.begin_list();
  subbuilder->append(1.1);
  subbuilder->append(2.2);
  subbuilder->append(3.3);
  builder.end_list();

  builder.begin_list();
  subbuilder->append(4.4);
  subbuilder->append(5.5);
  subbuilder->append(6.6);
  builder.end_list();

  // [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RegularArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"size\": 3, "
      "\"form_key\": \"node0\" "
  "}");

  double* ptr0 = new double[builder.content()->length()];
  builder.content()->to_buffers(ptr0);

  dump("node0", ptr0, builder.content()->length());
  std::cout<<std::endl;
}

void
test_Regular_size0() {
  auto builder = lb::Regular<0,
      lb::Numpy<initial, double>
  >();

  auto subbuilder = builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  // [[], []]

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"RegularArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"size\": 0, "
      "\"form_key\": \"node0\" "
  "}");

  double* ptr0 = new double[builder.content()->length()];
  builder.content()->to_buffers(ptr0);

  dump("node0", ptr0, builder.content()->length());
  std::cout<<std::endl;
}

int main(int /* argc */, char ** /* argv */) {
  test_Numpy_bool();
  test_Numpy_char();
  test_Numpy_int();
  test_Numpy_double();
  test_Numpy_complex();
  test_ListOffset();
  test_ListOffset_ListOffset();
  test_Record();
  test_EmptyRecord();
  test_ListOffset_Record();
  test_Record_Record();
  test_Record_nested();
  test_List();
  test_Indexed();
  test_IndexedOption();
  test_Empty();
  test_ListOffset_Empty();
  test_Unmasked();
  test_ByteMasked();
  test_BitMasked();
  test_Regular();
  test_Regular_size0();

  return 0;
}

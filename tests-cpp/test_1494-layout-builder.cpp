// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/LayoutBuilder.h"

namespace lb = awkward::LayoutBuilder;
using UserDefinedMap = std::map<std::size_t, std::string>;

static const char param[] = "";

static const unsigned initial = 10;

template <class NODE, class PRIMITIVE, class LENGTH>
void dump(NODE&& node, PRIMITIVE&& ptr, LENGTH&& length) {
  std::cout << node << ": ";
  for (size_t at = 0; at < length; at++) {
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

template<class PRIMITIVE>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<initial, PRIMITIVE>;

template<class BUILDER>
using ListOffsetBuilder = awkward::LayoutBuilder::ListOffset<initial, BUILDER>;

template<typename... BUILDERS>
using RecordBuilder = awkward::LayoutBuilder::Record<UserDefinedMap, BUILDERS...>;

void
test_Numpy_bool() {

  NumpyBuilder<bool> builder;

  builder.append(true);
  builder.append(false);
  builder.append(true);
  builder.append(true);

  // [True, False, True, True]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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
  NumpyBuilder<int64_t> builder;

  size_t data_size = 10;

  int64_t data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

  builder.extend(data, data_size);

 // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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
  NumpyBuilder<char> builder;

  builder.append('a');
  builder.append('b');
  builder.append('c');
  builder.append('d');

  // ['a', 'b', 'c', 'd']

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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
  NumpyBuilder<double> builder;

  size_t data_size = 9;

  double data[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  builder.extend(data, data_size);

  // [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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
  NumpyBuilder<std::complex<double>> builder;

  builder.append({1.1, 0.1});
  builder.append({2.2, 0.2});
  builder.append({3.3, 0.3});
  builder.append({4.4, 0.4});
  builder.append({5.5, 0.5});

  // [1.1 + 0.1j, 2.2 + 0.2j, 3.3 + 0.3j, 4.4 + 0.4j, 5.5 + 0.5j]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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
  ListOffsetBuilder<NumpyBuilder<double>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.append(1.1);
  subbuilder.append(2.2);
  subbuilder.append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.append(4.4);
  subbuilder.append(5.5);
  builder.end_list();

  // [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

  auto form = builder.form();
  std::cout << form << std::endl;
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

  double* ptr1 = new double[subbuilder.length()];
  subbuilder.to_buffers(ptr1);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, subbuilder.length());
  std::cout<<std::endl;
}

void
test_ListOffset_ListOffset() {

  ListOffsetBuilder<ListOffsetBuilder<NumpyBuilder<double>>> builder;

  auto& builder2 = builder.begin_list();

  auto& builder3 = builder2.begin_list();
  builder3.append(1.1);
  builder3.append(2.2);
  builder3.append(3.3);
  builder2.end_list();
  builder2.begin_list();
  builder2.end_list();
  builder.end_list();

  builder.begin_list();
  builder2.begin_list();
  builder3.append(4.4);
  builder3.append(5.5);
  builder2.end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder2.begin_list();
  builder3.append(6.6);
  builder2.end_list();
  builder2.begin_list();
  builder3.append(7.7);
  builder3.append(8.8);
  builder3.append(9.9);
  builder2.end_list();
  builder.end_list();

  // [
  //     [[1.1, 2.2, 3.3], []],
  //     [[4.4, 5.5]],
  //     [],
  //     [[6.6], [7.7, 8.8, 9.9]],
  // ]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 3);

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

  int64_t* ptr1 = new int64_t[builder2.length() + 1];
  builder2.to_buffers(ptr1);

  double* ptr2 = new double[builder3.length()];
  builder3.to_buffers(ptr2);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, builder2.length() + 1,
       "node2", ptr2, builder3.length());
  std::cout<<std::endl;
}

void
test_EmptyRecord() {
  lb::EmptyRecord<true> builder;

  builder.append();

  builder.extend(2);

  // [(), (), ()]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 0);

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
  // A user has to provide the field names by mapping
  // the enum values to std::strings:
  //
  using UserDefinedMap = std::map<std::size_t, std::string>;

  enum Field : std::size_t {one, two, three};

  UserDefinedMap fields_map({
    {Field::one, "one"},
    {Field::two, "two"},
    {Field::three, "three"}});

  RecordBuilder<
      lb::Field<Field::one, NumpyBuilder<double>>,
      lb::Field<Field::two, NumpyBuilder<int64_t>>,
      lb::Field<Field::three, NumpyBuilder<char>>
  > builder(fields_map);

  auto names = builder.field_names();
  for (auto i : names) {
    std::cout << "field name " << i << std::endl;
  }

  auto& one_builder = builder.field<Field::one>();
  auto& two_builder = builder.field<Field::two>();
  auto& three_builder = builder.field<Field::three>();

  three_builder.append('a');

  one_builder.append(3.3);
  two_builder.append(4);
  three_builder.append('b');

  // [
  //     {"one": 1.1, "two": 2, "three": 'a'},
  //     {"one": 3.3, "two": 4. "three": 'b'},
  // ]

  // assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  std::cout << names_nbytes.size();
  assert (names_nbytes.size() == 3);

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

  double* ptr0 = new double[one_builder.length()];
  one_builder.to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[two_builder.length()];
  two_builder.to_buffers(ptr1);

  char* ptr2 = new char[three_builder.length()];
  three_builder.to_buffers(ptr2);

  dump("node1", ptr0, one_builder.length(),
       "node2", ptr1, two_builder.length(),
       "node3", ptr2, three_builder.length());
  std::cout<<std::endl;
}

void
test_ListOffset_Record() {

  enum Field : int {x, y};

  UserDefinedMap fields_map({
    {Field::x, "x"},
    {Field::y, "y"}});

  ListOffsetBuilder<
      RecordBuilder<
          lb::Field<Field::x, NumpyBuilder<double>>,
          lb::Field<Field::y, ListOffsetBuilder<
              NumpyBuilder<int32_t>>
  >>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.set_field_names(fields_map);

  auto& x_builder = subbuilder.field<Field::x>();
  auto& y_builder = subbuilder.field<Field::y>();

  x_builder.append(1.1);
  auto& y_subbuilder = y_builder.begin_list();
  y_subbuilder.append(1);
  y_builder.end_list();

  x_builder.append(2.2);
  y_builder.begin_list();
  y_subbuilder.append(1);
  y_subbuilder.append(2);
  y_builder.end_list();

  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();

  x_builder.append(3.3);
  y_builder.begin_list();
  y_subbuilder.append(1);
  y_subbuilder.append(2);
  y_subbuilder.append(3);
  y_builder.end_list();

  builder.end_list();

  // [
  //     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
  //     [],
  //     [{"x": 3.3, "y": [1, 2, 3]}],
  // ]

  // assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 4);

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

  double* ptr1 = new double[x_builder.length()];
  x_builder.to_buffers(ptr1);

  int64_t* ptr2 = new int64_t[y_builder.length() + 1];
  y_builder.to_buffers(ptr2);

  int32_t* ptr3 = new int32_t[y_subbuilder.length()];
  y_subbuilder.to_buffers(ptr3);

  dump("node0", ptr0, builder.length() + 1,
       "node2", ptr1, x_builder.length(),
       "node3", ptr2, y_builder.length() + 1,
       "node4", ptr3, y_subbuilder.length());
  std::cout<<std::endl;
}

void
test_Record_Record()
{
  std::cout << "test_Record_Record()\n";

  using UserDefinedMap = std::map<std::size_t, std::string>;

  enum Field0 : std::size_t {x, y};

  UserDefinedMap fields_map0({
    {Field0::x, "x"},
    {Field0::y, "y"}});

  enum Field1 : std::size_t {u, v};

  UserDefinedMap fields_map1({
    {Field1::u, "u"},
    {Field1::v, "v"}});

  enum Field2 : std::size_t {w};

  UserDefinedMap fields_map2({
    {Field2::w, "w"}});

  RecordBuilder<
      lb::Field<Field0::x, RecordBuilder<
          lb::Field<Field1::u, NumpyBuilder<double>>,
          lb::Field<Field1::v, ListOffsetBuilder<NumpyBuilder<int64_t>>>>>,
      lb::Field<Field0::y, RecordBuilder<
          lb::Field<Field2::w, NumpyBuilder<char>>>>
  > builder;
  builder.set_field_names(fields_map0);

  auto& x_builder = builder.field<Field0::x>();
  x_builder.set_field_names(fields_map1);

  auto& y_builder = builder.field<Field0::y>();
  y_builder.set_field_names(fields_map2);

  auto& u_builder = x_builder.field<Field1::u>();
  auto& v_builder = x_builder.field<Field1::v>();

  auto& w_builder = y_builder.field<Field2::w>();


  u_builder.append(1.1);
  auto& v_subbuilder = v_builder.begin_list();
  v_subbuilder.append(1);
  v_subbuilder.append(2);
  v_subbuilder.append(3);
  v_builder.end_list();

  w_builder.append('a');

  u_builder.append(3.3);
  v_builder.begin_list();
  v_subbuilder.append(4);
  v_subbuilder.append(5);
  v_builder.end_list();

  w_builder.append('b');

  // [
  //     {"x": {"u": 1.1, "v": [1, 2, 3]}, "y": {"w": 'a'}},
  //     {"x": {"u": 3.3, "v": [4, 5]}, "y": {"w": 'b'}},
  // ]

  // assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 4);

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

  double* ptr0 = new double[u_builder.length()];
  u_builder.to_buffers(ptr0);

  int64_t* ptr1 = new int64_t[v_builder.length() + 1];
  v_builder.to_buffers(ptr1);

  int64_t* ptr2 = new int64_t[v_subbuilder.length()];
  v_subbuilder.to_buffers(ptr2);

  char* ptr3 = new char[w_builder.length()];
  w_builder.to_buffers(ptr3);

  dump("node2", ptr0, u_builder.length(),
       "node3", ptr1, v_builder.length() + 1,
       "node4", ptr2, v_subbuilder.length(),
       "node6", ptr3, w_builder.length());
  std::cout << std::endl;
}

void
test_Record_nested()
{
  std::cout << "test_Record_nested()\n";

  using UserDefinedMap = std::map<std::size_t, std::string>;

  enum Field0 : std::size_t {u, v, w};

  UserDefinedMap fields_map0({
    {Field0::u, "u"},
    {Field0::v, "v"},
    {Field0::w, "w"}});

  enum Field1 : std::size_t {i, j};

  UserDefinedMap fields_map1({
    {Field1::i, "i"},
    {Field1::j, "j"}});

  RecordBuilder<
      lb::Field<Field0::u, ListOffsetBuilder<
          RecordBuilder<
              lb::Field<Field1::i, NumpyBuilder<double>>,
              lb::Field<Field1::j, ListOffsetBuilder<
                  NumpyBuilder<int64_t>>>>>>,
      lb::Field<Field0::v, NumpyBuilder<int64_t>>,
      lb::Field<Field0::w, NumpyBuilder<double>>
  > builder;
  builder.set_field_names(fields_map0);

  auto& u_builder = builder.field<Field0::u>();
  auto& v_builder = builder.field<Field0::v>();
  auto& w_builder = builder.field<Field0::w>();

  auto& u_subbuilder = u_builder.begin_list();
  u_subbuilder.set_field_names(fields_map1);

  auto& i_builder = u_subbuilder.field<Field1::i>();
  auto& j_builder = u_subbuilder.field<Field1::j>();

  i_builder.append(1.1);
  auto& j_subbuilder = j_builder.begin_list();
  j_subbuilder.append(1);
  j_subbuilder.append(2);
  j_subbuilder.append(3);
  j_builder.end_list();

  u_builder.end_list();

  v_builder.append(-1);
  w_builder.append(3.3);

  u_builder.begin_list();

  i_builder.append(2.2);
  j_builder.begin_list();
  j_subbuilder.append(4);
  j_subbuilder.append(5);
  j_builder.end_list();

  u_builder.end_list();

  v_builder.append(-2);
  w_builder.append(4.4);

  // [
  //     {"u": [{"i": 1.1, "j": [1, 2, 3]}], "v": -1, "w": 3.3},
  //     {"u": [{"i": 2.2, "j": [4, 5]}], "v": -2, "w": 4.4},
  // ]

  // assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 6);

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

  int64_t* ptr0 = new int64_t[u_builder.length() + 1];
  u_builder.to_buffers(ptr0);

  double* ptr1 = new double[i_builder.length()];
  i_builder.to_buffers(ptr1);

  int64_t* ptr2 = new int64_t[j_builder.length() + 1];
  j_builder.to_buffers(ptr2);

  int64_t* ptr3 = new int64_t[j_subbuilder.length()];
  j_subbuilder.to_buffers(ptr3);

  int64_t* ptr4 = new int64_t[v_builder.length()];
  v_builder.to_buffers(ptr4);

  double* ptr5 = new double[w_builder.length()];
  w_builder.to_buffers(ptr5);

  dump("node1", ptr0, u_builder.length() + 1,
       "node3", ptr1, i_builder.length(),
       "node4", ptr2, j_builder.length() + 1,
       "node5", ptr3, j_subbuilder.length(),
       "node6", ptr4, v_builder.length(),
       "node7", ptr5, w_builder.length());
  std::cout << "DONE!" << std::endl;
}

void
test_List() {
  lb::List<initial, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.append(1.1);
  subbuilder.append(2.2);
  subbuilder.append(3.3);
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.append(4.4);
  subbuilder.append(5.5);
  builder.end_list();

  builder.begin_list();
  subbuilder.append(6.6);

  builder.end_list();

  builder.begin_list();
  subbuilder.append(7.7);
  subbuilder.append(8.8);
  subbuilder.append(9.9);
  builder.end_list();

  // [
  //     [1.1, 2.2, 3.3],
  //     [],
  //     [4.4, 5.5],
  //     [6.6],
  //     [7.7, 8.8, 9.9],
  // ]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 3);

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
  double* ptr3 = new double[subbuilder.length()];
  subbuilder.to_buffers(ptr3);
  dump("node0", ptr1, builder.length(),
       "     ", ptr2, builder.length(),
       "node1", ptr3, subbuilder.length());
  std::cout<<std::endl;
}

void
test_Indexed() {
  lb::Indexed<initial, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.append_index();
  subbuilder.append(1.1);

  builder.append_index();
  subbuilder.append(2.2);

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder.extend(data, 3);

  // [1.1, 2.2, 3.3, 4.4, 5.5]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

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

  double* ptr1 = new double[builder.content().length()];
  builder.content().to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content().length());
  std::cout<<std::endl;
}

void
test_IndexedOption() {
  lb::IndexedOption<initial, NumpyBuilder<double>> builder;

  auto& subbuilder = builder.append_index();
  subbuilder.append(1.1);

  builder.append_null();

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder.extend(data, 3);

  builder.extend_null(2);

  // [1.1, None, 3.3, 4.4, 5.5, None, None]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

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

  double* ptr1 = new double[builder.content().length()];
  builder.content().to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content().length());
  std::cout<<std::endl;
}

void
test_Empty() {
  lb::Empty builder;

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 0);

  auto form = builder.form();

  assert (form ==
  "{ "
      "\"class\": \"EmptyArray\" "
  "}");
}

void
test_ListOffset_Empty() {
  ListOffsetBuilder<
      ListOffsetBuilder<lb::Empty
  >> builder;

  builder.begin_list();
  builder.end_list();

  auto& subbuilder = builder.begin_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  //  [[], [[], [], []], [[], []], [], [[]]]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

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

  int64_t* ptr1 = new int64_t[subbuilder.length() + 1];
  subbuilder.to_buffers(ptr1);

  dump("node0", ptr0, builder.length() + 1,
       "node1", ptr1, subbuilder.length() + 1);
  std::cout<<std::endl;
}

void
test_Unmasked() {
  lb::Unmasked<NumpyBuilder<int64_t>> builder;

  auto& subbuilder = builder.append_valid();
  subbuilder.append(11);
  subbuilder.append(22);
  subbuilder.append(33);
  subbuilder.append(44);
  subbuilder.append(55);

  // [11, 22, 33, 44, 55]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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

  int64_t* ptr0 = new int64_t[builder.content().length()];
  builder.content().to_buffers(ptr0);

  dump("node0", ptr0, builder.content().length());
  std::cout<<std::endl;
}

void
test_ByteMasked() {
  lb::ByteMasked<true, initial,
      NumpyBuilder<double>
  > builder;

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);

  builder.append_null();
  subbuilder.append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);

  builder.extend_null(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder.append(-1000);  // have to supply a "dummy" value
  }

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

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

  double* ptr1 = new double[builder.content().length()];
  builder.content().to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content().length());
  std::cout<<std::endl;
}

void
test_BitMasked() {
  lb::BitMasked<true, true, initial,
      NumpyBuilder<double>
  > builder;

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);

  builder.append_null();
  subbuilder.append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);

  builder.extend_null(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder.append(-1000);  // have to supply a "dummy" value
  }

  builder.append_valid();
  subbuilder.append(8);

  builder.append_valid();
  subbuilder.append(9);

  builder.append_valid();
  subbuilder.append(10);

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000, 8, 9, 10]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 2);

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

  double* ptr1 = new double[builder.content().length()];
  builder.content().to_buffers(ptr1);

  dump("node0", ptr0, builder.length(),
       "node1", ptr1, builder.content().length());
  std::cout<<std::endl;
}

void
test_Regular() {
  lb::Regular<3,
      NumpyBuilder<double>
  > builder;

  auto& subbuilder = builder.begin_list();
  subbuilder.append(1.1);
  subbuilder.append(2.2);
  subbuilder.append(3.3);
  builder.end_list();

  builder.begin_list();
  subbuilder.append(4.4);
  subbuilder.append(5.5);
  subbuilder.append(6.6);
  builder.end_list();

  // [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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

  double* ptr0 = new double[builder.content().length()];
  builder.content().to_buffers(ptr0);

  dump("node0", ptr0, builder.content().length());
  std::cout<<std::endl;
}

void
test_Regular_size0() {
  lb::Regular<0,
      NumpyBuilder<double>
  > builder;

  auto& subbuilder = builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  // [[], []]

  assert (builder.is_valid() == true);

  std::map<std::string, int64_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert (names_nbytes.size() == 1);

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

  double* ptr0 = new double[builder.content().length()];
  builder.content().to_buffers(ptr0);

  dump("node0", ptr0, builder.content().length());
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

// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include "awkward/LayoutBuilder.h"

template <class NODE, class PRIMITIVE, class LENGTH>
void dump(std::ostringstream& out, NODE&& node, PRIMITIVE&& ptr, LENGTH&& length) {
  out << node << ": ";
  for (size_t i = 0; i < length; i++) {
    out << +ptr[i] << " ";
  }
  out << std::endl;
}

template<class NODE, class PRIMITIVE, class LENGTH, class ... Args>
void dump(std::ostringstream& out, NODE&& node, PRIMITIVE&& ptr, LENGTH&& length, Args&&...args)
{
    dump(out, node, ptr, length);
    dump(out, args...);
}

std::map<std::string, void*>
empty_buffers(std::map<std::string, size_t> &names_nbytes) {
  std::map<std::string, void*> buffers = {};
  for(const auto& it : names_nbytes) {
    auto* ptr = new uint8_t[it.second];
    buffers[it.first] = (void*)ptr;
  }
  return buffers;
}

void
clear_buffers(std::map<std::string, void*> &buffers) {
  for(const auto& it : buffers) {
    delete[] (uint8_t*)it.second;
  }
  buffers.clear();
}

using UserDefinedMap = std::map<std::size_t, std::string>;

template<class PRIMITIVE>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<PRIMITIVE>;

template<class PRIMITIVE, class BUILDER>
using ListOffsetBuilder = awkward::LayoutBuilder::ListOffset<PRIMITIVE, BUILDER>;

using EmptyBuilder = awkward::LayoutBuilder::Empty;

template<class... BUILDERS>
using RecordBuilder = awkward::LayoutBuilder::Record<UserDefinedMap, BUILDERS...>;

template<std::size_t field_name, class BUILDER>
using RecordField = awkward::LayoutBuilder::Field<field_name, BUILDER>;

template<class... BUILDERS>
using TupleBuilder = awkward::LayoutBuilder::Tuple<BUILDERS...>;

template <unsigned SIZE, class BUILDER>
using RegularBuilder = awkward::LayoutBuilder::Regular<SIZE, BUILDER>;

template<class PRIMITIVE, class BUILDER>
using IndexedBuilder = awkward::LayoutBuilder::Indexed<PRIMITIVE, BUILDER>;

template<class PRIMITIVE, class BUILDER>
using IndexedOptionBuilder = awkward::LayoutBuilder::IndexedOption<PRIMITIVE, BUILDER>;

template<class BUILDER>
using UnmaskedBuilder = awkward::LayoutBuilder::Unmasked<BUILDER>;

template<bool VALID_WHEN, class BUILDER>
using ByteMaskedBuilder = awkward::LayoutBuilder::ByteMasked<VALID_WHEN, BUILDER>;

template<bool VALID_WHEN, bool LSB_ORDER, class BUILDER>
using BitMaskedBuilder = awkward::LayoutBuilder::BitMasked<VALID_WHEN, LSB_ORDER, BUILDER>;

template<class... BUILDERS>
using UnionBuilder8_U32 = awkward::LayoutBuilder::Union<int8_t, uint32_t, BUILDERS...>;

template<class... BUILDERS>
using UnionBuilder8_64 = awkward::LayoutBuilder::Union<int8_t, int64_t, BUILDERS...>;

template<class PRIMITIVE>
using StringBuilder = awkward::LayoutBuilder::String<PRIMITIVE>;

void
test_Numpy_bool() {

  NumpyBuilder<bool> builder;
  assert(builder.length() == 0);

  builder.append(true);
  builder.append(false);
  builder.append(true);
  builder.append(true);
  assert(builder.length() == 4);

  // [True, False, True, True]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node0-data", (bool*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(bool));

  std::string check{"node0-data: 1 0 1 1 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"bool\", "
      "\"form_key\": \"node0\" "
  "}");

  assert(names_nbytes["node0-data"] == sizeof(bool[4]));
  assert(builder.length() == 4);

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Numpy_int() {
  NumpyBuilder<int64_t> builder;
  assert(builder.length() == 0);

  int64_t data[10] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
  auto data_length = sizeof(data)/sizeof(int64_t);

  builder.extend(data, data_length);
  assert(builder.length() == 10);

 // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node0-data", (int64_t*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(int64_t));

  std::string check{"node0-data: -5 -4 -3 -2 -1 0 1 2 3 4 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"int64\", "
      "\"form_key\": \"node0\" "
  "}");

  assert(names_nbytes["node0-data"] == sizeof(data));
  assert(builder.length() == data_length);

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Numpy_char() {
  NumpyBuilder<char> builder;
  assert(builder.length() == 0);

  builder.append('a');
  builder.append('b');
  builder.append('c');
  builder.append('d');
  assert(builder.length() == 4);

  // ['a', 'b', 'c', 'd']

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node0-data", (char*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(char));

  std::string check{"node0-data: 97 98 99 100 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"char\", "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Numpy_double() {
  NumpyBuilder<double> builder;
  assert(builder.length() == 0);

  builder.append(1.1);
  builder.append(2.2);
  assert(builder.length() == 2);

  size_t data_size = 3;

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend(data, data_size);
  assert(builder.length() == 5);

  // [1.1, 2.2, 3.3, 4.4, 5.5]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node0-data", (double*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(double));

  std::string check{"node0-data: 1.1 2.2 3.3 4.4 5.5 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"float64\", "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Numpy_complex() {
  NumpyBuilder<std::complex<double>> builder;
  assert(builder.length() == 0);

  builder.append({1.1, 0.1});
  builder.append({2.2, 0.2});
  builder.append({3.3, 0.3});
  builder.append({4.4, 0.4});
  builder.append({5.5, 0.5});

  // [1.1 + 0.1j, 2.2 + 0.2j, 3.3 + 0.3j, 4.4 + 0.4j, 5.5 + 0.5j]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node0-data", (std::complex<double>*)buffers["node0-data"], names_nbytes["node0-data"]/sizeof(std::complex<double>));

  std::string check{"node0-data: (1.1,0.1) (2.2,0.2) (3.3,0.3) (4.4,0.4) (5.5,0.5) \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"complex128\", "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_ListOffset() {
  ListOffsetBuilder<int64_t, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

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

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node0-offsets: 0 3 3 5 \n"
                    "node1-data: 1.1 2.2 3.3 4.4 5.5 \n"};
  assert(out.str().compare(check) == 0);

  assert(names_nbytes["node0-offsets"] == sizeof(int64_t[4]));
  assert(names_nbytes["node1-data"] == sizeof(double[5]));

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_ListOffset_ListOffset() {
  ListOffsetBuilder<int64_t,
      ListOffsetBuilder<int32_t, NumpyBuilder<double>>
  > builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.begin_list();

  auto& subsubbuilder = subbuilder.begin_list();
  subsubbuilder.append(1.1);
  subsubbuilder.append(2.2);
  subsubbuilder.append(3.3);
  subbuilder.end_list();
  subbuilder.begin_list();
  subbuilder.end_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.begin_list();
  subsubbuilder.append(4.4);
  subsubbuilder.append(5.5);
  subbuilder.end_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  subbuilder.begin_list();
  subsubbuilder.append(6.6);
  subbuilder.end_list();
  subbuilder.begin_list();
  subsubbuilder.append(7.7);
  subsubbuilder.append(8.8);
  subsubbuilder.append(9.9);
  subbuilder.end_list();
  builder.end_list();

  // [
  //     [[1.1, 2.2, 3.3], []],
  //     [[4.4, 5.5]],
  //     [],
  //     [[6.6], [7.7, 8.8, 9.9]],
  // ]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node1-offsets", (int32_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int32_t),
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double));

  std::string check{"node0-offsets: 0 2 3 3 5 \n"
                    "node1-offsets: 0 3 3 5 6 9 \n"
                    "node2-data: 1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i32\", "
          "\"content\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node2\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Empty() {
  EmptyBuilder builder;
  assert(builder.length() == 0);

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 0);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  assert(builder.form() ==
  "{ "
      "\"class\": \"EmptyArray\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_ListOffset_Empty() {
  ListOffsetBuilder<int64_t,
      ListOffsetBuilder<int64_t, EmptyBuilder>
  > builder;
  assert(builder.length() == 0);

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

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node1-offsets", (int64_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int64_t));

  std::string check{"node0-offsets: 0 0 3 5 5 6 \n"
                    "node1-offsets: 0 0 0 0 0 0 0 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Record()
{
  enum Field : std::size_t {one, two, three};

  UserDefinedMap fields_map({
    {Field::one, "one"},
    {Field::two, "two"},
    {Field::three, "three"}});

    RecordBuilder<
        RecordField<Field::one, NumpyBuilder<double>>,
        RecordField<Field::two, NumpyBuilder<int64_t>>,
        RecordField<Field::three, NumpyBuilder<char>>
    > builder(fields_map);
  assert(builder.length() == 0);

  std::vector<std::string> fields {"one", "two", "three"};

  auto names = builder.fields();

  for (size_t i = 0; i < names.size(); i++) {
    assert(names[i] == fields[i]);
  }

  auto& one_builder = builder.content<Field::one>();
  auto& two_builder = builder.content<Field::two>();
  auto& three_builder = builder.content<Field::three>();

  three_builder.append('a');

  one_builder.append(1.1);
  one_builder.append(3.3);

  two_builder.append(2);
  two_builder.append(4);

  std::string error;
  assert(builder.is_valid(error) == false);
  assert(error == "Record node0 has field \"three\" length 1 that differs from the first length 2\n");

  three_builder.append('b');

  // [
  //     {"one": 1.1, "two": 2, "three": 'a'},
  //     {"one": 3.3, "two": 4, "three": 'b'},
  // ]

  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double),
       "node2-data", (int64_t*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(int64_t),
       "node3-data", (char*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(char));

  std::string check{"node1-data: 1.1 3.3 \n"
                    "node2-data: 2 4 \n"
                    "node3-data: 97 98 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_ListOffset_Record() {
  enum Field : std::size_t {x, y};

  UserDefinedMap fields_map({
    {Field::x, "x"},
    {Field::y, "y"}});

  ListOffsetBuilder<int64_t,
      RecordBuilder<
          RecordField<Field::x, NumpyBuilder<double>>,
          RecordField<Field::y, ListOffsetBuilder<int64_t,
              NumpyBuilder<int32_t>>
  >>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.begin_list();
  subbuilder.set_fields(fields_map);

  auto& x_builder = subbuilder.content<Field::x>();
  auto& y_builder = subbuilder.content<Field::y>();

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

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 4);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-offsets", (int64_t*)buffers["node0-offsets"], names_nbytes["node0-offsets"]/sizeof(int64_t),
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
       "node3-offsets", (int64_t*)buffers["node3-offsets"], names_nbytes["node3-offsets"]/sizeof(int64_t),
       "node4-data", (int32_t*)buffers["node4-data"], names_nbytes["node4-data"]/sizeof(int32_t));

  std::string check{"node0-offsets: 0 2 2 3 \n"
                    "node2-data: 1.1 2.2 3.3 \n"
                    "node3-offsets: 0 1 3 6 \n"
                    "node4-data: 1 1 2 1 2 3 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Record_Record()
{
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
      RecordField<Field0::x, RecordBuilder<
          RecordField<Field1::u, NumpyBuilder<double>>,
          RecordField<Field1::v, ListOffsetBuilder<int64_t,
              NumpyBuilder<int64_t>>>>>,
      RecordField<Field0::y, RecordBuilder<
          RecordField<Field2::w, NumpyBuilder<char>>>>
  > builder;
  builder.set_fields(fields_map0);
  assert(builder.length() == 0);

  auto& x_builder = builder.content<Field0::x>();
  x_builder.set_fields(fields_map1);

  auto& y_builder = builder.content<Field0::y>();
  y_builder.set_fields(fields_map2);

  auto& u_builder = x_builder.content<Field1::u>();
  auto& v_builder = x_builder.content<Field1::v>();

  auto& w_builder = y_builder.content<Field2::w>();

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

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 4);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
       "node3-offsets", (int64_t*)buffers["node3-offsets"], names_nbytes["node3-offsets"]/sizeof(int64_t),
       "node4-data", (int64_t*)buffers["node4-data"], names_nbytes["node4-data"]/sizeof(int64_t),
       "node6-data", (char*)buffers["node6-data"], names_nbytes["node6-data"]/sizeof(char));

  std::string check{"node2-data: 1.1 3.3 \n"
                    "node3-offsets: 0 3 5 \n"
                    "node4-data: 1 2 3 4 5 \n"
                    "node6-data: 97 98 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Record_nested()
{
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
      RecordField<Field0::u, ListOffsetBuilder<int64_t,
          RecordBuilder<
              RecordField<Field1::i, NumpyBuilder<double>>,
              RecordField<Field1::j, ListOffsetBuilder<int64_t,
                  NumpyBuilder<int64_t>>>
      >>>,
      RecordField<Field0::v, NumpyBuilder<int64_t>>,
      RecordField<Field0::w, NumpyBuilder<double>>
  > builder;
  builder.set_fields(fields_map0);
  assert(builder.length() == 0);

  auto& u_builder = builder.content<Field0::u>();
  auto& v_builder = builder.content<Field0::v>();
  auto& w_builder = builder.content<Field0::w>();

  auto& u_subbuilder = u_builder.begin_list();
  u_subbuilder.set_fields(fields_map1);

  auto& i_builder = u_subbuilder.content<Field1::i>();
  auto& j_builder = u_subbuilder.content<Field1::j>();

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

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 6);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node1-offsets", (int64_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int64_t),
       "node3-data", (double*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(double),
       "node4-offsets", (int64_t*)buffers["node4-offsets"], names_nbytes["node4-offsets"]/sizeof(int64_t),
       "node5-data", (int64_t*)buffers["node5-data"], names_nbytes["node5-data"]/sizeof(int64_t),
       "node6-data", (int64_t*)buffers["node6-data"], names_nbytes["node6-data"]/sizeof(int64_t),
       "node7-data", (double*)buffers["node7-data"], names_nbytes["node7-data"]/sizeof(double));

  std::string check{"node1-offsets: 0 1 2 \n"
                    "node3-data: 1.1 2.2 \n"
                    "node4-offsets: 0 3 5 \n"
                    "node5-data: 1 2 3 4 5 \n"
                    "node6-data: -1 -2 \n"
                    "node7-data: 3.3 4.4 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Tuple_Numpy_ListOffset() {
  TupleBuilder<
      NumpyBuilder<double>,
      ListOffsetBuilder<int64_t, NumpyBuilder<int32_t>>
  > builder;
  assert(builder.length() == 0);

  std::string error;
  assert(builder.is_valid(error) == true);

  auto& subbuilder_one = builder.content<0>();
  subbuilder_one.append(1.1);
  auto& subbuilder_two = builder.content<1>();
  auto& subsubbuilder = subbuilder_two.begin_list();
  subsubbuilder.append(1);
  subbuilder_two.end_list();

  assert(builder.is_valid(error) == true);

  subbuilder_one.append(2.2);
  subbuilder_two.begin_list();
  subsubbuilder.append(1);
  subsubbuilder.append(2);
  subbuilder_two.end_list();

  assert(builder.is_valid(error) == true);

  subbuilder_one.append(3.3);
  subbuilder_two.begin_list();
  subsubbuilder.append(1);
  subsubbuilder.append(2);
  subsubbuilder.append(3);
  subbuilder_two.end_list();

  // [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]

  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double),
       "node2-offsets", (int64_t*)buffers["node2-offsets"], names_nbytes["node2-offsets"]/sizeof(int64_t),
       "node3-data", (int32_t*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(int32_t));

  std::string check{"node1-data: 1.1 2.2 3.3 \n"
                    "node2-offsets: 0 1 3 6 \n"
                    "node3-data: 1 1 2 1 2 3 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"RecordArray\", "
      "\"contents\": ["
        "{ "
            "\"class\": \"NumpyArray\", "
            "\"primitive\": \"float64\", "
            "\"form_key\": \"node1\" "
        "}, "
        "{ "
            "\"class\": \"ListOffsetArray\", "
            "\"offsets\": \"i64\", "
            "\"content\": { "
                "\"class\": \"NumpyArray\", "
                "\"primitive\": \"int32\", "
                "\"form_key\": \"node3\" "
            "}, "
            "\"form_key\": \"node2\" "
        "}], "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Regular() {
  RegularBuilder<3, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

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

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node1-data: 1.1 2.2 3.3 4.4 5.5 6.6 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Regular_size0() {
  RegularBuilder<0, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

  builder.begin_list();
  builder.end_list();

  builder.begin_list();
  builder.end_list();

  // [[], []]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node1-data: \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Indexed() {
  IndexedBuilder<uint32_t, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.append_index();
  subbuilder.append(1.1);

  builder.append_index();
  subbuilder.append(2.2);

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_index(3);
  subbuilder.extend(data, 3);

  // [1.1, 2.2, 3.3, 4.4, 5.5]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-index", (uint32_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(uint32_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node0-index: 0 1 2 3 4 \n"
                    "node1-data: 1.1 2.2 3.3 4.4 5.5 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"IndexedArray\", "
      "\"index\": \"u32\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_IndexedOption() {
  IndexedOptionBuilder<int32_t, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);

  builder.append_invalid();

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);

  builder.extend_invalid(2);

  // [1.1, None, 3.3, 4.4, 5.5, None, None]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-index", (int32_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(int32_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node0-index: 0 -1 1 2 3 -1 -1 \n"
                    "node1-data: 1.1 3.3 4.4 5.5 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
      "\"index\": \"i32\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_IndexedOption_Record() {
  enum Field : std::size_t {x, y};

  UserDefinedMap fields_map({
    {Field::x, "x"},
    {Field::y, "y"}});

  IndexedOptionBuilder<int64_t, RecordBuilder<
      RecordField<Field::x, NumpyBuilder<double>>,
      RecordField<Field::y, NumpyBuilder<int64_t>>
  >> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.append_valid();
  subbuilder.set_fields(fields_map);

  auto& x_builder = subbuilder.content<Field::x>();
  auto& y_builder = subbuilder.content<Field::y>();

  x_builder.append(1.1);
  y_builder.append(2);

  builder.append_invalid();

  builder.append_valid();
  x_builder.append(3.3);
  y_builder.append(4);

  // [
  //   {x: 1.1, y: 2},
  //   {},
  //   {x: 3.3, y: 4},
  // ]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 3);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-index", (int64_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(int64_t),
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
       "node3-data", (int64_t*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(int64_t));

  std::string check{"node0-index: 0 -1 1 \n"
                    "node2-data: 1.1 3.3 \n"
                    "node3-data: 2 4 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
            "\"class\": \"RecordArray\", "
            "\"contents\": { "
                "\"x\": { "
                    "\"class\": \"NumpyArray\", "
                    "\"primitive\": \"float64\", "
                    "\"form_key\": \"node2\" "
                "}, "
                "\"y\": { "
                    "\"class\": \"NumpyArray\", "
                    "\"primitive\": \"int64\", "
                    "\"form_key\": \"node3\" "
                "} "
            "}, "
            "\"form_key\": \"node1\" "
        "}, "
        "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Unmasked() {
  UnmaskedBuilder<NumpyBuilder<int64_t>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.content();
  subbuilder.append(11);
  subbuilder.append(22);
  subbuilder.append(33);
  subbuilder.append(44);
  subbuilder.append(55);

  // [11, 22, 33, 44, 55]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 1);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out, "node1-data", (int64_t*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(int64_t));

  std::string check{"node1-data: 11 22 33 44 55 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"UnmaskedArray\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"int64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_ByteMasked() {
  ByteMaskedBuilder<true, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);

  builder.append_invalid();
  subbuilder.append(-1000); // have to supply a "dummy" value

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);

  builder.extend_invalid(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder.append(-1000);  // have to supply a "dummy" value
  }

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-mask", (int8_t*)buffers["node0-mask"], names_nbytes["node0-mask"]/sizeof(int8_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node0-mask: 1 0 1 1 1 0 0 \n"
                    "node1-data: 1.1 -1000 3.3 4.4 5.5 -1000 -1000 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_BitMasked() {
  BitMaskedBuilder<true, true, NumpyBuilder<double>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.append_valid();
  subbuilder.append(1.1);
  assert(builder.length() == 1);

  builder.append_invalid();
  subbuilder.append(-1000); // have to supply a "dummy" value
  assert(builder.length() == 2);

  double data[3] = {3.3, 4.4, 5.5};

  builder.extend_valid(3);
  subbuilder.extend(data, 3);
  assert(builder.length() == 5);

  builder.extend_invalid(2);
  for (size_t i = 0; i < 2; i++) {
    subbuilder.append(-1000);  // have to supply a "dummy" value
  }
  assert(builder.length() == 7);

  builder.append_valid();
  subbuilder.append(8);
  assert(builder.length() == 8);

  builder.append_valid();
  subbuilder.append(9);
  assert(builder.length() == 9);

  builder.append_valid();
  subbuilder.append(10);
  assert(builder.length() == 10);

  // [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000, 8, 9, 10]

  std::string error;
  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes.size() == 2);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-mask", (uint8_t*)buffers["node0-mask"], names_nbytes["node0-mask"]/sizeof(uint8_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double));

  std::string check{"node0-mask: 157 3 \n"
                    "node1-data: 1.1 -1000 3.3 4.4 5.5 -1000 -1000 8 9 10 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
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

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_Union8_U32_Numpy_ListOffset() {
  UnionBuilder8_U32<
      NumpyBuilder<double>,
      ListOffsetBuilder<int64_t, NumpyBuilder<int32_t>>
  > builder;
  assert(builder.length() == 0);

  std::string error;
  assert(builder.is_valid(error) == true);

  auto &subbuilder_one = builder.append_content<0>();
  subbuilder_one.append(1.1);

  assert(builder.is_valid(error) == true);

  auto& subbuilder_two = builder.append_content<1>();
  auto& subsubbuilder = subbuilder_two.begin_list();
  subsubbuilder.append(1);
  subsubbuilder.append(2);
  subbuilder_two.end_list();

  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);

  assert(names_nbytes.size() == 5);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-tags", (int8_t*)buffers["node0-tags"], names_nbytes["node0-tags"]/sizeof(int8_t),
       "node0-index", (uint32_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(uint32_t),
       "node1-data", (double*)buffers["node1-data"], names_nbytes["node1-data"]/sizeof(double),
       "node2-offsets", (int64_t*)buffers["node2-offsets"], names_nbytes["node2-offsets"]/sizeof(int64_t),
       "node3-data", (int32_t*)buffers["node3-data"], names_nbytes["node3-data"]/sizeof(int32_t));

  std::string check{"node0-tags: 0 1 \n"
                    "node0-index: 0 0 \n"
                    "node1-data: 1.1 \n"
                    "node2-offsets: 0 2 \n"
                    "node3-data: 1 2 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"UnionArray\", "
      "\"tags\": \"i8\", "
      "\"index\": \"u32\", "
      "\"contents\": ["
      "{ "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"float64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "{ "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i64\", "
          "\"content\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"int32\", "
              "\"form_key\": \"node3\" "
          "}, "
          "\"form_key\": \"node2\" "
      "}], "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
  test_Union8_64_ListOffset_Record() {
  enum Field : std::size_t {x, y};

  UserDefinedMap fields_map({
    {Field::x, "x"},
    {Field::y, "y"}});

  UnionBuilder8_64<
      ListOffsetBuilder<int64_t, NumpyBuilder<double>>,
      RecordBuilder<
          RecordField<Field::x, NumpyBuilder<int64_t>>,
          RecordField<Field::y, NumpyBuilder<char>>
  >> builder;
  assert(builder.length() == 0);

  std::string error;
  assert(builder.is_valid(error) == true);

  auto& subbuilder_one = builder.append_content<0>();
  auto& subsubbuilder = subbuilder_one.begin_list();
  subsubbuilder.append(1.1);
  subsubbuilder.append(3.3);
  subbuilder_one.end_list();

  assert(builder.is_valid(error) == true);

  auto &subbuilder_two = builder.append_content<1>();
  subbuilder_two.set_fields(fields_map);

  auto& x_builder = subbuilder_two.content<Field::x>();
  auto& y_builder = subbuilder_two.content<Field::y>();

  x_builder.append(1);
  y_builder.append('a');

  assert(builder.is_valid(error) == true);

  builder.append_content<0>();
  subbuilder_one.begin_list();
  subsubbuilder.append(5.5);
  subbuilder_one.end_list();

  assert(builder.is_valid(error) == true);

  builder.append_content<1>();
  x_builder.append(2);
  y_builder.append('b');

  assert(builder.is_valid(error) == true);

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);

  assert(names_nbytes.size() == 6);

  auto buffers = empty_buffers(names_nbytes);
  builder.to_buffers(buffers);

  std::ostringstream out;
  dump(out,
       "node0-tags", (int8_t*)buffers["node0-tags"], names_nbytes["node0-tags"]/sizeof(int8_t),
       "node0-index", (int64_t*)buffers["node0-index"], names_nbytes["node0-index"]/sizeof(int64_t),
       "node1-offsets", (int64_t*)buffers["node1-offsets"], names_nbytes["node1-offsets"]/sizeof(int64_t),
       "node2-data", (double*)buffers["node2-data"], names_nbytes["node2-data"]/sizeof(double),
       "node4-data", (int64_t*)buffers["node4-data"], names_nbytes["node4-data"]/sizeof(int64_t),
       "node5-data", (char*)buffers["node5-data"], names_nbytes["node5-data"]/sizeof(char));

  std::string check{"node0-tags: 0 1 0 1 \n"
                    "node0-index: 0 0 1 1 \n"
                    "node1-offsets: 0 2 3 \n"
                    "node2-data: 1.1 3.3 5.5 \n"
                    "node4-data: 1 2 \n"
                    "node5-data: 97 98 \n"};
  assert(out.str().compare(check) == 0);

  assert(builder.form() ==
  "{ "
      "\"class\": \"UnionArray\", "
      "\"tags\": \"i8\", "
      "\"index\": \"i64\", "
      "\"contents\": ["
      "{ "
          "\"class\": \"ListOffsetArray\", "
          "\"offsets\": \"i64\", "
          "\"content\": { "
              "\"class\": \"NumpyArray\", "
              "\"primitive\": \"float64\", "
              "\"form_key\": \"node2\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "{ "
          "\"class\": \"RecordArray\", "
          "\"contents\": { "
              "\"x\": { "
                  "\"class\": \"NumpyArray\", "
                  "\"primitive\": \"int64\", "
                  "\"form_key\": \"node4\" "
              "}, "
              "\"y\": { "
                  "\"class\": \"NumpyArray\", "
                  "\"primitive\": \"char\", "
                  "\"form_key\": \"node5\" "
              "} "
          "}, "
          "\"form_key\": \"node3\" "
      "}], "
      "\"form_key\": \"node0\" "
  "}");

  clear_buffers(buffers);
  builder.clear();
  assert(builder.length() == 0);
}

void
test_char_form() {
  NumpyBuilder<uint8_t> builder;

  builder.set_parameters("\"__array__\": \"char\"");

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"uint8\", "
      "\"parameters\": { "
          "\"__array__\": \"char\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  builder.clear();
  assert(builder.length() == 0);
}

void
test_string_form() {
  ListOffsetBuilder<int64_t, NumpyBuilder<uint8_t>> builder;
  assert(builder.length() == 0);

  auto& subbuilder = builder.content();

  builder.set_parameters("\"__array__\": \"string\"");

  subbuilder.set_parameters("\"__array__\": \"char\"");

  assert(builder.form() ==
  "{ "
      "\"class\": \"ListOffsetArray\", "
      "\"offsets\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"uint8\", "
          "\"parameters\": { "
              "\"__array__\": \"char\" "
          "}, "
          "\"form_key\": \"node1\" "
      "}, "
      "\"parameters\": { "
          "\"__array__\": \"string\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  builder.clear();
  assert(builder.length() == 0);
}

void
test_categorical_form() {
  IndexedOptionBuilder<int64_t, NumpyBuilder<int64_t>> builder;
  assert(builder.length() == 0);

  builder.set_parameters("\"__array__\": \"categorical\"");

  assert(builder.form() ==
  "{ "
      "\"class\": \"IndexedOptionArray\", "
      "\"index\": \"i64\", "
      "\"content\": { "
          "\"class\": \"NumpyArray\", "
          "\"primitive\": \"int64\", "
          "\"form_key\": \"node1\" "
      "}, "
      "\"parameters\": { "
          "\"__array__\": \"categorical\" "
      "}, "
      "\"form_key\": \"node0\" "
  "}");

  builder.clear();
  assert(builder.length() == 0);
}


void test_string_builder() {
  StringBuilder<int64_t> builder;
  assert(builder.length() == 0);

  builder.append("one");
  builder.append("two");
  builder.append("three");

  assert(builder.length() == 3);
}

void test_list_string_builder() {
  ListOffsetBuilder<int64_t, StringBuilder<int64_t>> builder;
  assert(builder.length() == 0);

  builder.begin_list();
  builder.content().append("one");
  builder.content().append("two");
  builder.content().append("three");
  builder.end_list();

  builder.begin_list();
  builder.content().append("four");
  builder.content().append("five");
  builder.end_list();
}

int main(int /* argc */, char ** /* argv */) {
  test_Numpy_bool();
  test_Numpy_int();
  test_Numpy_char();
  test_Numpy_double();
  test_Numpy_complex();
  test_ListOffset();
  test_ListOffset_ListOffset();
  test_Empty();
  test_ListOffset_Empty();
  test_Record();
  test_ListOffset_Record();
  test_Record_Record();
  test_Record_nested();
  test_Tuple_Numpy_ListOffset();
  test_Regular();
  test_Regular_size0();
  test_Indexed();
  test_IndexedOption();
  test_IndexedOption_Record();
  test_Unmasked();
  test_ByteMasked();
  test_BitMasked();
  test_Union8_U32_Numpy_ListOffset();
  test_Union8_64_ListOffset_Record();
  test_char_form();
  test_string_form();
  test_categorical_form();
  test_string_builder();
  test_list_string_builder();

  return 0;
}

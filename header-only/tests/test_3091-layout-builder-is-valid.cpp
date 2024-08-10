// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include "awkward/LayoutBuilder.h"

// if BUG_FIXED is false, the tests confirm the presence of the bugs in is_valid
const bool BUG_FIXED=1;

template<class PRIMITIVE, class BUILDER>
using IndexedBuilder = awkward::LayoutBuilder::Indexed<PRIMITIVE, BUILDER>;

template<class PRIMITIVE, class BUILDER>
using IndexedOptionBuilder = awkward::LayoutBuilder::IndexedOption<PRIMITIVE, BUILDER>;

template<class PRIMITIVE>
using StringBuilder = awkward::LayoutBuilder::String<PRIMITIVE>;

void
test_Indexed_categorical() {
  IndexedBuilder<uint32_t, StringBuilder<double>> builder;
  assert(builder.length() == 0);
  builder.set_parameters("\"__array__\": \"categorical\"");

  auto& subbuilder = builder.append_index(0);
  builder.append_index(1);

  subbuilder.append("zero");
  subbuilder.append("one");

  std::string error;
  assert(builder.is_valid(error) == true);

  // index and content could have different lengths
  builder.append_index(1);
  assert(builder.is_valid(error) == BUG_FIXED);
}

void
test_Indexed_categorical_invalid_index() {
  IndexedBuilder<uint32_t, StringBuilder<double>> builder;
  assert(builder.length() == 0);
  builder.set_parameters("\"__array__\": \"categorical\"");

  auto& subbuilder = builder.append_index(0);
  builder.append_index(1);

  subbuilder.append("zero");
  subbuilder.append("one");

  std::string error;
  assert(builder.is_valid(error) == true);

  // index should be less than the length of content
  subbuilder.append("two");
  builder.append_index(9);
  bool assertion = builder.is_valid(error) == !BUG_FIXED;
  // std::cout << error << std::endl;
  assert(assertion);
}

void
test_IndexedOption_categorical() {
  IndexedOptionBuilder<int32_t, StringBuilder<double>> builder;
  assert(builder.length() == 0);
  builder.set_parameters("\"__array__\": \"categorical\"");

  builder.append_invalid();
  auto& subbuilder = builder.append_valid(1);
  subbuilder.append("zero");
  builder.append_valid(1);
  subbuilder.append("one");

  std::string error;
  bool assertion = builder.is_valid(error);
  // std::cout << error << std::endl;
  assert(assertion);

  // index and content could have different lengths
  builder.append_valid(1);
  builder.append_valid(1);
  builder.append_valid(1);
  assert(builder.is_valid(error) == BUG_FIXED);
}

void
test_IndexedOption_categorical_invalid_index() {
  IndexedOptionBuilder<int32_t, StringBuilder<double>> builder;
  assert(builder.length() == 0);
  builder.set_parameters("\"__array__\": \"categorical\"");

  builder.append_invalid();
  auto& subbuilder = builder.append_valid(1);
  subbuilder.append("zero");
  builder.append_valid(1);
  subbuilder.append("one");

  std::string error;
  bool assertion = builder.is_valid(error);
  // std::cout << error << std::endl;
  assert(assertion);

  // index should be less than the length of content
  builder.append_valid(9);
  subbuilder.append("two");
  assertion = builder.is_valid(error) == !BUG_FIXED;
  // std::cout << error << std::endl;
  assert(assertion);
}

void
test_Indexed_empty() {
  IndexedBuilder<uint32_t, StringBuilder<double>> builder;
  assert(builder.length() == 0);

  // empty indexed builder should be valid
  std::string error;
  assert(builder.is_valid(error));
}

void
test_IndexedOption_empty() {
  IndexedOptionBuilder<uint32_t, StringBuilder<double>> builder;
  assert(builder.length() == 0);

  // empty indexed builder should be valid
  std::string error;
  assert(builder.is_valid(error));

  // content has length 0 but still should be valid
  builder.append_invalid();
  assert(builder.is_valid(error));
}

int main(int /* argc */, char ** /* argv */) {
  test_Indexed_categorical();
  test_Indexed_categorical_invalid_index();
  test_IndexedOption_categorical();
  test_IndexedOption_categorical_invalid_index();
  test_IndexedOption_empty();
  test_Indexed_empty();

  return 0;
}

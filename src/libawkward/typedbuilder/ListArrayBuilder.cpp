// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ListArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ListArray.h"

namespace awkward {

  ///
  /// FIXME: implement Form morfing
  /// ListForm to ListOffsetForm
  ///
  ListArrayBuilder::ListArrayBuilder(const ListFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) { }

  const std::string
  ListArrayBuilder::classname() const {
    return "ListArrayBuilder";
  }

  const ContentPtr
  ListArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // if(content_ != nullptr) {
    //   Index64 starts(reinterpret_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
    //   Index64 stops(reinterpret_pointer_cast<int64_t>(data_), length_, length_, kernel::lib::cpu);
    //   return std::make_shared<ListArray64>(Identities::none(),
    //                                        form_.get()->parameters(),
    //                                        starts,
    //                                        stops,
    //                                        content_.get()->snapshot(outputs));
    // }
    // else {
    //   throw std::invalid_argument(
    //     std::string("Form of a ") + classname()
    //     + std::string(" needs another Form as its content")
    //     + FILENAME(__LINE__));
    // }
    return nullptr;
  }

  const FormPtr
  ListArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ListArrayBuilder::vm_output() const {
    return std::string("\n");
  }

  const std::string
  ListArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  ListArrayBuilder::vm_func_name() const {
    std::string out;
    out.append(*form_key_)
      .append("-")
      .append("list");
    return out;
  }

}

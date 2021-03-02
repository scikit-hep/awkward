// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/RecordArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/RecordArray.h"

namespace awkward {

  ///
  RecordArrayBuilder::RecordArrayBuilder(const RecordFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    for (auto const& content : form.get()->contents()) {
      contents_.push_back(TypedArrayBuilder::formBuilderFromA(content));
    }

    for (auto const& content : contents_) {
      vm_output_.append(content.get()->vm_output());
    }
    for (auto const& content : contents_) {
      vm_from_stack_.append(content.get()->vm_from_stack());
    }

    vm_func_name_ = std::string(*form_key_).append("-record");

    for (auto const& content : contents_) {
      vm_func_.append(content.get()->vm_func());
    }
    vm_func_.append(": ")
      .append(vm_func_name_);

    for (auto const& content : contents_) {
      vm_func_.append("\n    ").append(content.get()->vm_func_name())
        .append(" pause");
    }
    // Remove the last pause
    vm_func_.erase(vm_func_.end() - 6, vm_func_.end());
    vm_func_.append("\n;\n\n");
  }

  const std::string
  RecordArrayBuilder::classname() const {
    return "RecordArrayBuilder";
  }

  const ContentPtr
  RecordArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    ContentPtrVec contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      contents.push_back(contents_[i].get()->snapshot(outputs));
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         form_.get()->parameters(),
                                         contents,
                                         form_.get()->recordlookup());
  }

  const FormPtr
  RecordArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  RecordArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  RecordArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  RecordArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  RecordArrayBuilder::vm_from_stack() const {
    return vm_from_stack_;
  }

}

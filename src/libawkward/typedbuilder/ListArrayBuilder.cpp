// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ListArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ListArray.h"

namespace awkward {

  ///
  ListArrayBuilder::ListArrayBuilder(const ListFormPtr& form,
                                     const std::string attribute,
                                     const std::string partition)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    vm_output_data_ = std::string("part")
      .append(partition_).append("-")
      .append(*form_key_).append("-")
      .append(attribute_);

    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->starts()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(std::to_string(static_cast<utype>(state::begin_list)))
      .append(" <> if").append("\n")
      .append(std::to_string(TypedArrayBuilder::next_error_id())).append(" err ! err @ halt").append("\n")
      .append("then").append("\n")
      .append("\n")
      .append("0").append("\n")
      .append("begin").append("\n")
      .append("pause").append("\n")
      .append("dup ")
      .append(std::to_string(static_cast<utype>(state::end_list)))
      .append(" = if").append("\n")
      .append("drop").append("\n")
      .append(vm_output_data_).append(" +<- stack").append("\n")
      .append("exit").append("\n")
      .append("else").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("1+").append("\n")
      .append("then").append("\n")
      .append("again").append("\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");

    vm_error_.append(content_.get()->vm_error());
    vm_error_.append("s\"ListArray Builder needs begin_list\"").append("\n");
 }

  const std::string
  ListArrayBuilder::classname() const {
    return "ListArrayBuilder";
  }

  const ContentPtr
  ListArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      if (form_.get()->starts() == Index::Form::i32) {
        Index32 offsets = search->second.get()->toIndex32();
        Index32 starts = util::make_starts(offsets);
        Index32 stops = util::make_stops(offsets);
        return std::make_shared<ListArray32>(Identities::none(),
                                             form_.get()->parameters(),
                                             starts,
                                             stops,
                                             content_.get()->snapshot(outputs));
      }
      else if (form_.get()->starts() == Index::Form::u32) {
        IndexU32 offsets = search->second.get()->toIndexU32();
        IndexU32 starts = util::make_starts(offsets);
        IndexU32 stops = util::make_stops(offsets);
        return std::make_shared<ListArrayU32>(Identities::none(),
                                              form_.get()->parameters(),
                                              starts,
                                              stops,
                                              content_.get()->snapshot(outputs));
      }
      else if (form_.get()->starts() == Index::Form::i64) {
        Index64 offsets = search->second.get()->toIndex64();
        Index64 starts = util::make_starts(offsets);
        Index64 stops = util::make_stops(offsets);
        return std::make_shared<ListArray64>(Identities::none(),
                                             form_.get()->parameters(),
                                             starts,
                                             stops,
                                             content_.get()->snapshot(outputs));
      }
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs offsets")
        + FILENAME(__LINE__));
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
    return vm_output_;
  }

  const std::string
  ListArrayBuilder::vm_func_name() const {
    return vm_func_;
  }

  const std::string
  ListArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  ListArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  ListArrayBuilder::vm_error() const {
    return vm_error_;
  }

}

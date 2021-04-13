// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/UnionArrayBuilder.cpp", line)

#include "awkward/typedbuilder/UnionArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/UnionArray.h"

namespace awkward {

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form,
                                       const std::string attribute,
                                       const std::string partition)
    : form_(form),
      tag_(0),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition) {
    vm_func_type_ = std::to_string(static_cast<utype>(state::tag));

    for (auto const& content : form.get()->contents()) {
      contents_.push_back(TypedArrayBuilder::formBuilderFromA(content));
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
      vm_error_.append(contents_.back().get()->vm_error());
    }

    vm_output_tags_ = std::string("part")
      .append(partition_)
      .append("-")
      .append(*form_key_)
      .append("-tags");

    vm_output_.append("output ")
      .append(vm_output_tags_)
      .append(" ")
      .append(index_form_to_name(form_.get()->tags()))
      .append("\n");

    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);
    for (auto const& content : contents_) {
      vm_func_.append(content.get()->vm_func());
    }
    vm_func_.append(": ")
      .append(vm_func_name_).append("\n")
      .append(vm_func_type())
      .append(" = if").append("\n");

    vm_func_.append("0 data seek\n")
      .append("data ").append(index_form_to_vm_format(form_.get()->tags()))
      .append("-> stack dup ").append(vm_output_tags_).append(" <- stack\n");

    int64_t tag = 0;
    int64_t contents_size = (int64_t)contents_.size();
    bool drop = true;
    for (auto const& content : contents_) {
      drop = (tag < contents_size) ? true : false;
      if (drop) {
        vm_func_.append("dup ");
      }

      vm_func_.append(std::to_string(tag++))
        .append(" = if").append("\n");

      if (drop) {
        vm_func_.append("drop").append("\n");
        drop = false;
      }

      vm_func_.append("pause").append("\n");

      vm_func_.append(content.get()->vm_func_name())
        .append("\n")
        .append("exit").append("\n");

      vm_func_.append("then").append("\n");
    }
    vm_func_.append(std::to_string(TypedArrayBuilder::next_error_id()))
    .append(" err ! err @ halt").append("\n")
    .append("then\n;\n\n");

    vm_error_.append("s\" Union Array Builder error\"\n");
  }

  const std::string
  UnionArrayBuilder::classname() const {
    return "UnionArrayBuilder";
  }

  const ContentPtr
  UnionArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search_tags = outputs.find(vm_output_tags_);
    if (search_tags != outputs.end()) {
      Index8 tags(std::static_pointer_cast<int8_t>(search_tags->second.get()->ptr()),
                  0,
                  search_tags->second.get()->len(),
                  kernel::lib::cpu);

      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->snapshot(outputs));
      }

      int64_t lentags = tags.length();

      if (form_.get()->index() == Index::Form::i32) {
        Index32 current(lentags);
        Index32 outindex(lentags);
        struct Error err = kernel::UnionArray_regular_index<int8_t, int32_t>(
          kernel::lib::cpu,   // DERIVE
          outindex.data(),
          current.data(),
          lentags,
          tags.data(),
          lentags);
        util::handle_error(err, "UnionArray", nullptr);

        return UnionArray8_32(Identities::none(),
                              util::Parameters(),
                              tags,
                              outindex,
                              contents).simplify_uniontype(false, false);

      }
      else if (form_.get()->index() == Index::Form::u32) {
        IndexU32 current(lentags);
        IndexU32 outindex(lentags);
        struct Error err = kernel::UnionArray_regular_index<int8_t, uint32_t>(
          kernel::lib::cpu,   // DERIVE
          outindex.data(),
          current.data(),
          lentags,
          tags.data(),
          lentags);
        util::handle_error(err, "UnionArray", nullptr);

        return UnionArray8_U32(Identities::none(),
                               util::Parameters(),
                               tags,
                               outindex,
                               contents).simplify_uniontype(false, false);
      }
      else if (form_.get()->index() == Index::Form::i64) {
        Index64 current(lentags);
        Index64 outindex(lentags);
        struct Error err = kernel::UnionArray_regular_index<int8_t, int64_t>(
          kernel::lib::cpu,   // DERIVE
          outindex.data(),
          current.data(),
          lentags,
          tags.data(),
          lentags);
        util::handle_error(err, "UnionArray", nullptr);

        return UnionArray8_64(Identities::none(),
                              util::Parameters(),
                              tags,
                              outindex,
                              contents).simplify_uniontype(false, false);
      }
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs tags and index ")
        + FILENAME(__LINE__));
  }

  const FormPtr
  UnionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  UnionArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  UnionArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  UnionArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  UnionArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  UnionArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  UnionArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  UnionArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  UnionArrayBuilder::tag(int8_t x) {
    tag_ = x;
  }

  void
  UnionArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->boolean(x, builder);
  }

  void
  UnionArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->int64(x, builder);
  }

  void
  UnionArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->float64(x, builder);
  }

  void
  UnionArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->complex(x, builder);
  }

  void
  UnionArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->bytestring(x, builder);
  }

  void
  UnionArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->string(x, builder);
  }

  void
  UnionArrayBuilder::begin_list(TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->begin_list(builder);
  }

  void
  UnionArrayBuilder::end_list(TypedArrayBuilder* builder) {
    contents_[(size_t)tag_].get()->end_list(builder);
  }

}

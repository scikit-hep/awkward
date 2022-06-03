// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/UnionArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/UnionArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  template <typename T, typename I>
  UnionArrayBuilder<T, I>::UnionArrayBuilder(const std::vector<FormBuilderPtr<T, I>>& contents,
                                             const util::Parameters& parameters,
                                             const std::string& form_key,
                                             const std::string& form_tags,
                                             const std::string& form_index,
                                             const std::string attribute,
                                             const std::string partition)
    : parameters_(parameters),
      form_key_(form_key),
      tag_(0),
      form_index_(index_form_to_name(form_index)) {
    vm_func_type_ = std::to_string(static_cast<utype>(state::tag));

    for (auto const& content : contents) {
      contents_.push_back(content);
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
      vm_error_.append(contents_.back().get()->vm_error());
    }

    vm_output_tags_ = std::string("part")
      .append(partition)
      .append("-")
      .append(form_key)
      .append("-tags");

    vm_output_.append("output ")
      .append(vm_output_tags_)
      .append(" ")
      .append(index_form_to_name(form_tags))
      .append(" ");

    vm_func_name_ = std::string(form_key).append("-").append(attribute);
    for (auto const& content : contents_) {
      vm_func_.append(content.get()->vm_func());
    }
    vm_func_.append(": ")
      .append(vm_func_name_).append(" ")
      .append(vm_func_type())
      .append(" = if ");

    vm_func_.append("0 data seek data ")
      .append(index_form_to_vm_format(form_tags))
      .append("-> stack dup ").append(vm_output_tags_).append(" <- stack ");

    int64_t tag = 0;
    int64_t contents_size = (int64_t)contents_.size();
    bool drop = true;
    for (auto const& content : contents_) {
      drop = (tag < contents_size) ? true : false;
      if (drop) {
        vm_func_.append("dup ");
      }

      vm_func_.append(std::to_string(tag++))
        .append(" = if ");

      if (drop) {
        vm_func_.append("drop ");
        drop = false;
      }

      vm_func_.append("pause ");

      vm_func_.append(content.get()->vm_func_name())
        .append(" exit then ");
    }
    vm_func_.append(std::to_string(LayoutBuilder<T, I>::next_error_id()))
    .append(" err ! err @ halt then ; ");

    vm_error_.append("s\" Union Array Builder error\" ");
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::classname() const {
    return "UnionArrayBuilder";
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::to_buffers(
    BuffersContainer& container,
    const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_tags());
    if (search != outputs.end()) {
      auto length = (ssize_t)search->second.get()->len();

      auto tags = search->second.get()->toIndex8();

      Index64 current(length);
      Index64 index(length);
      struct Error err = kernel::UnionArray_regular_index<int8_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        current.data(),
        length,
        tags.data(),
        length);
      util::handle_error(err, "UnionArray", nullptr);

      container.copy_buffer(form_key() + "-tags",
                            tags.ptr().get(),
                            (int64_t)(length * (ssize_t)sizeof(int8_t)));

      container.copy_buffer(form_key() + "-index",
                            index.ptr().get(),
                            (int64_t)(index.length() * (ssize_t)sizeof(int64_t)));

      std::stringstream out;
      out << "{\"class\": \"UnionArray\", \"tags\": \"i8\", \"index\": \"i64\", \"contents\": [";
      for (size_t i = 0;  i < contents().size();  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << contents_[i].get()->to_buffers(container, outputs);
      }
      out << "], \"form_key\": \"" << form_key() + "\"}";
      return out.str();
    }
    throw std::invalid_argument(
      std::string("Snapshot of a ") + classname()
      + std::string(" needs tags ")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_output() const {
    return vm_output_;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_output_data() const {
    return vm_output_data_;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_func() const {
    return vm_func_;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_func_name() const {
    return vm_func_name_;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_func_type() const {
    return vm_func_type_;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::tag(int8_t x) {
    tag_ = x;
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->boolean(x, builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->int64(x, builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->float64(x, builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->complex(x, builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->bytestring(x, builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->string(x, builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->begin_list(builder);
  }

  template <typename T, typename I>
  void
  UnionArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
    contents_[(size_t)tag_].get()->end_list(builder);
  }

  template class EXPORT_TEMPLATE_INST UnionArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST UnionArrayBuilder<int64_t, int32_t>;

}

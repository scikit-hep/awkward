// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/IndexedArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/IndexedArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  /// @brief
  template <typename T, typename I>
  IndexedArrayBuilder<T, I>::IndexedArrayBuilder(FormBuilderPtr<T, I> content,
                                                 const util::Parameters& parameters,
                                                 const std::string& form_key,
                                                 const std::string& form_index,
                                                 bool is_categorical,
                                                 const std::string attribute,
                                                 const std::string partition)
    : content_(content),
      parameters_(parameters),
      is_categorical_(is_categorical),
      form_index_(form_index),
      form_key_(form_key) {
    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_func_name_ = std::string(form_key).append("-")
      .append(attribute).append("-")
      .append(form_index);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(form_index)
      .append(" ")
      .append(content_.get()->vm_output())
      .append("variable index ");

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name())
      .append(" dup ").append(std::to_string(static_cast<utype>(state::index)))
      .append(" = if drop ")
      .append(vm_output_data_).append(" <- stack else 1 index +! index @ 1- ")
      .append(vm_output_data_).append(" <- stack ")
      .append(content_.get()->vm_func_name())
      .append(" then ; ");

    vm_error_ = content_.get()->vm_error();
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::classname() const {
    return "IndexedArrayBuilder";
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::to_buffers(
    BuffersContainer& container,
    const ForthOutputBufferMap& outputs) const {

    auto search = outputs.find(vm_output_data());
    if (search != outputs.end()) {
      if (form_index() == "int32") {
        container.copy_buffer(form_key() + "-index",
          search->second.get()->ptr().get(),
          (int32_t)((ssize_t)search->second.get()->len() * (ssize_t)sizeof(int32_t)));

        return "{\"class\": \"IndexedArray\", \"index\": \"i32\", \"content\": "
          + content().get()->to_buffers(container, outputs) + ", "
          + this->parameters_as_string(parameters_) + " \"form_key\": \""
          + form_key() + "\"}";

      } else if (form_index() == "int64") {
        container.copy_buffer(form_key() + "-index",
          search->second.get()->ptr().get(),
          (int64_t)((ssize_t)search->second.get()->len() * (ssize_t)sizeof(int64_t)));

        return "{\"class\": \"IndexedArray\", \"index\": \"i64\", \"content\": "
          + content().get()->to_buffers(container, outputs) + ", "
          + this->parameters_as_string(parameters_) + " \"form_key\": \""
          + form_key() + "\"}";

      } else {
        throw std::invalid_argument(
            std::string("Snapshot of a ") + classname()
            + std::string(" index ") + form_index()
            + std::string(" is not supported yet. ")
            + FILENAME(__LINE__));
      }

    }
    throw std::invalid_argument(
      std::string("Snapshot of a ") + classname()
      + std::string(" needs an index ")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_output() const {
    return vm_output_;
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_output_data() const {
    return vm_output_data_;
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_func() const {
    return vm_func_;
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_func_name() const {
    return vm_func_name_;
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_func_type() const {
    return vm_func_type_;
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  template <typename T, typename I>
  const std::string
  IndexedArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of(x, data)) {
        return;
      }
    }
    content_.get()->boolean(x, builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of(x, data)) {
        return;
      }
    }
    content_.get()->int64(x, builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of(x, data)) {
        return;
      }
    }
    content_.get()->float64(x, builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of(x, data)) {
        return;
      }
    }
    content_.get()->complex(x, builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    if (is_categorical_) {
      throw std::runtime_error(
        std::string("IndexedArrayBuilder categorical 'bytestring' is not implemented yet")
        + FILENAME(__LINE__));
    }
    content_.get()->bytestring(x, builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    if (is_categorical_) {
      throw std::runtime_error(
        std::string("IndexedArrayBuilder categorical 'string' is not implemented yet")
        + FILENAME(__LINE__));
    }
    content_.get()->string(x, builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
    content_.get()->begin_list(builder);
  }

  template <typename T, typename I>
  void
  IndexedArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
    content_.get()->end_list(builder);
  }

  template class EXPORT_TEMPLATE_INST IndexedArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST IndexedArrayBuilder<int64_t, int32_t>;

}

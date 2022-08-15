// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LayoutBuilder_H_
#define AWKWARD_LayoutBuilder_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/forth/ForthMachine.h"
#include "awkward/layoutbuilder/FormBuilder.h"

#include <complex>

namespace awkward {

  using ForthOutputBufferMap = std::map<std::string, std::shared_ptr<ForthOutputBuffer>>;

  const std::string
    index_form_to_name(const std::string& form_index);

  const std::string
    index_form_to_vm_format(const std::string& form_index);

  enum class state : std::int32_t {
    int64 = 0,
    float64 = 1,
    begin_list = 2,
    end_list = 3,
    boolean = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    uint8 = 8,
    uint16 = 9,
    uint32 = 10,
    uint64 = 11,
    float16 = 12,
    float32 = 13,
    float128 = 14,
    complex64 = 15,
    complex128 = 16,
    complex256 = 17,
    null = 18,
    index = 19,
    tag = 20,
    datetime64 = 21,
    timedelta64 = 22
  };
  using utype = std::underlying_type<state>::type;

  const std::string
    primitive_to_state(const std::string& name);

  const std::string
    primitive_to_vm_format(const std::string& name);

  /// @class LayoutBuilder
  ///
  /// @brief User interface to the FormBuilder system: the LayoutBuilder is a
  /// fixed reference while the FormBuilder subclass instances change in
  /// response to accumulating data.
  template <typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL LayoutBuilder {
  public:
    /// @brief Creates an LayoutBuilder from a full set of parameters.
    ///
    /// @param form The Form that defines the Array to be build.
    /// @param initial The Array builder initial.
    /// @param vm_init If 'true' the Virtual Machine is instantiated on
    /// construction. If 'false' an external Virtial Machine must be connected
    /// to the builder. The flag is used for debugging.
    LayoutBuilder(const std::string& json_form,
                  const int64_t initial,
                  bool vm_init = true);

    const std::string&
      json_form() const {
        return json_form_;
      }

    /// @brief Copy the current snapshot into the BuffersContainer and
    /// return a Form as a std::string (JSON).
    const std::string
      to_buffers(BuffersContainer& container) const;

    /// @brief Connects a Virtual Machine if it was not initialized before.
    void
      connect(const std::shared_ptr<ForthMachineOf<T, I>>& vm);

    /// @brief Prints debug information from the Virtual Machine stack.
    void
      debug_step() const;

    /// @brief Returns an AwkwardForth source code generated from the 'Form' and
    /// passed to the 'ForthMachine' virtual machine.
    const std::string
      vm_source() const;

    /// @brief
    const std::shared_ptr<ForthMachineOf<T, I>>
      vm() const;

    /// @brief Current length of the accumulated array.
    int64_t
      length() const;

    /// @brief
    void
      pre_snapshot() const;

    /// @brief Adds a `null` value to the accumulated data.
    void
      null();

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x);

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x);

    /// @brief Adds a real value `x` to the accumulated data.
    void
      float64(double x);

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x);

    /// @brief Adds an unencoded, null-terminated bytestring value `x` to the
    /// accumulated data.
    void
      bytestring(const char* x);

    /// @brief Adds an unencoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      bytestring(const char* x, int64_t length);

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x);

    /// @brief Adds a UTF-8 encoded, null-terminated bytestring value `x` to
    /// the accumulated data.
    void
      string(const char* x);

    /// @brief Adds a UTF-8 encoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      string(const char* x, int64_t length);

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x);

    /// @brief Begins building a nested list.
    ///
    /// The first 'beginlist' puts AwkwardForth VM into a state that expects
    /// another 'beginlist' or 'endlist'.
    /// The second puts the VM into a state that expects 'int64', etc.
    /// or 'endlist'.
    void
      begin_list();

    /// @brief Begins building a nested list.
    void
      add_begin_list();

    /// @brief Ends a nested list.
    void
      end_list();

    /// @brief Ends a nested list.
    void
      add_end_list();

    /// @brief Sets the pointer to a given tag `tag`; the next
    /// command will fill that slot.
    void
      tag(int8_t tag);

    /// @brief Issues an 'index' vm command. The value 'x' is pushed to
    /// the VM stack, it is not added to the accumulated data, e.g.
    /// the VM output buffer.
    ///
    /// This is used to build a 'categorical' array.
    void
      index(int64_t x);

    /// @brief Finds an index of a data in a VM output buffer.
    /// This is used to build a 'categorical' array.
    template <typename D>
    bool
      find_index_of(D x, const std::string& vm_output_data) {
        auto const& outputs = vm_.get()->outputs();
        auto search = outputs.find(vm_output_data);
        if (search != outputs.end()) {
          auto data = std::static_pointer_cast<D>(search->second.get()->ptr());
          auto size = search->second.get()->len();
          for (int64_t i = 0; i < size; i++) {
            if (data.get()[i] == x) {
              index(i);
              return true;
            }
          }
        }
        return false;
      }

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      add_bool(bool x);

    /// @brief Adds an int64_t value `x` to the accumulated data.
    void
      add_int64(int64_t x);

    /// @brief Adds a double value `x` to the accumulated data.
    void
      add_double(double x);

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      add_complex(std::complex<double> x);

    /// @brief Adds a string value `x` to the accumulated data.
    void
      add_string(const std::string& x);

    /// @brief Generates next unique ID
    static int64_t
      next_id();

    /// @brief Generates a user-defined error ID
    static int64_t
      next_error_id();

    /// @brief Resume Virtual machine run.
    void
      resume() const;

    // @brief Root node of the FormBuilder tree.
    const FormBuilderPtr<T, I> builder() const { return builder_; }

  protected:
    /// @brief A unique ID to use when Form nodes do not have Form key
    /// defined.
    static int64_t
      next_node_id;

    /// @brief An error ID to be used to generate a user 'halt' message.
    static int64_t
      error_id;

  private:
    /// @brief Generates an Array builder from a Form.
    FormBuilderPtr<T, I>
      form_builder_from_json(const std::string& json_form);

    /// @brief
    template <typename JSON>
    FormBuilderPtr<T, I>
      from_json(const JSON& json_doc);

    /// @ brief Initialise Layout Builder from a JSON Form.
    void
      initialise_builder(const std::string& json_form);

    /// @ brief Initialise Virtual machine.
    void
      initialise();

    /// @brief Place data of a type 'T' to the VM output buffer.
    template <typename D>
    void
      set_data(D x);

    /// @brief The Form that defines the Array to be build.
    const std::string json_form_;

    /// See #initial.
    int64_t initial_;

    /// @brief Root node of the FormBuilder tree.
    FormBuilderPtr<T, I> builder_;

    /// @brief Virtual machine.
    std::shared_ptr<ForthMachineOf<T, I>> vm_;

    /// @brief Virtual machine input buffers.
    std::map<std::string, std::shared_ptr<ForthInputBuffer>> vm_inputs_map_;

    /// @brief Input data label.
    std::string vm_input_data_;

    /// @brief Virtual machine source code.
    std::string vm_source_;

    /// @brief Virtual machine errors to ignore.
    std::set<util::ForthError> ignore_;

    /// @brief Virtual machine output buffers.
    ForthOutputBufferMap vm_outputs_map_;

  };

  using LayoutBuilder32 = LayoutBuilder<int32_t, int32_t>;
  using LayoutBuilder64 = LayoutBuilder<int64_t, int32_t>;

}

#endif // AWKWARD_LayoutBuilder_H_

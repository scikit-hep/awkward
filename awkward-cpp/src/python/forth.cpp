// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/forth.cpp", line)

#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "awkward/python/util.h"
#include "awkward/python/content.h"
#include "awkward/python/forth.h"

template <typename T, typename I>
py::object maybe_throw(const ak::ForthMachineOf<T, I>& self,
                       ak::util::ForthError err,
                       bool raise_user_halt,
                       bool raise_recursion_depth_exceeded,
                       bool raise_stack_underflow,
                       bool raise_stack_overflow,
                       bool raise_read_beyond,
                       bool raise_seek_beyond,
                       bool raise_skip_beyond,
                       bool raise_rewind_beyond,
                       bool raise_division_by_zero,
                       bool raise_varint_too_big,
                       bool raise_text_number_missing,
                       bool raise_quoted_string_missing,
                       bool raise_enumeration_missing) {
  std::set<ak::util::ForthError> ignore;
  if (!raise_user_halt) {
    ignore.insert(ak::util::ForthError::user_halt);
  }
  if (!raise_recursion_depth_exceeded) {
    ignore.insert(ak::util::ForthError::recursion_depth_exceeded);
  }
  if (!raise_stack_underflow) {
    ignore.insert(ak::util::ForthError::stack_underflow);
  }
  if (!raise_stack_overflow) {
    ignore.insert(ak::util::ForthError::stack_overflow);
  }
  if (!raise_read_beyond) {
    ignore.insert(ak::util::ForthError::read_beyond);
  }
  if (!raise_seek_beyond) {
    ignore.insert(ak::util::ForthError::seek_beyond);
  }
  if (!raise_skip_beyond) {
    ignore.insert(ak::util::ForthError::skip_beyond);
  }
  if (!raise_rewind_beyond) {
    ignore.insert(ak::util::ForthError::rewind_beyond);
  }
  if (!raise_division_by_zero) {
    ignore.insert(ak::util::ForthError::division_by_zero);
  }
  if (!raise_varint_too_big) {
    ignore.insert(ak::util::ForthError::varint_too_big);
  }
  if (!raise_text_number_missing) {
    ignore.insert(ak::util::ForthError::text_number_missing);
  }
  if (!raise_quoted_string_missing) {
    ignore.insert(ak::util::ForthError::quoted_string_missing);
  }
  if (!raise_enumeration_missing) {
    ignore.insert(ak::util::ForthError::enumeration_missing);
  }
  self.maybe_throw(err, ignore);

  switch (err) {
    case ak::util::ForthError::none:
      return py::none();
    case ak::util::ForthError::not_ready:
      return py::str("not ready");
    case ak::util::ForthError::is_done:
      return py::str("is done");
    case ak::util::ForthError::user_halt:
      return py::str("user halt");
    case ak::util::ForthError::recursion_depth_exceeded:
      return py::str("recursion depth exceeded");
    case ak::util::ForthError::stack_underflow:
      return py::str("stack underflow");
    case ak::util::ForthError::stack_overflow:
      return py::str("stack overflow");
    case ak::util::ForthError::read_beyond:
      return py::str("read beyond");
    case ak::util::ForthError::seek_beyond:
      return py::str("seek beyond");
    case ak::util::ForthError::skip_beyond:
      return py::str("skip beyond");
    case ak::util::ForthError::rewind_beyond:
      return py::str("rewind beyond");
    case ak::util::ForthError::division_by_zero:
      return py::str("division by zero");
    case ak::util::ForthError::varint_too_big:
      return py::str("varint too big");
    case ak::util::ForthError::text_number_missing:
      return py::str("text number missing");
    case ak::util::ForthError::quoted_string_missing:
      return py::str("quoted string missing");
    case ak::util::ForthError::enumeration_missing:
      return py::str("enumeration missing");
    default:
      throw std::invalid_argument(
          std::string("unrecognized ForthError: ")
          + std::to_string((int64_t)err) + FILENAME(__LINE__));
  }
}


template <typename T>
py::object capsule_for_shared_pointer(std::shared_ptr<T> ptr) {
  /**
   * Return a py::capsule that destructs the given pointer on GC
   */
  return py::capsule(new auto(ptr), [](void* p) {
    delete reinterpret_cast<decltype(ptr)*>(p);
  });
}


py::object output_buffer_to_numpy(std::shared_ptr<ak::ForthOutputBuffer> output) {
  auto ptr = output->ptr();
  // Hold long-lived shared_ptr, and delete when out of scope
  // `new auto(ptr)` creates long-lived pointer to a shared_ptr that shares
  // ownership with the shared-ptr `ptr`
  auto lifetime = capsule_for_shared_pointer(ptr);
  return py::array(py::dtype(ak::util::dtype_to_format(output->dtype())),
                        output->len(),
                        ptr.get(),
                        lifetime);
}


template <typename T, typename I>
py::object machine_bytecodes_at_to_python_content(std::shared_ptr<ak::ForthMachineOf<T, I>> machine, int64_t index) {
  // Single-copy into shared-ptr for offsets and bytecodes
  const auto offsets = machine->bytecodes_offsets();
  const auto length = (int64_t)offsets.size() - 1;
  const auto bytecodes_holder = std::make_shared<std::vector<I>>(std::move(machine->bytecodes()));

  // Build capsules which release their usage of the held memory on GC
  auto bytecodes_capsule = capsule_for_shared_pointer(bytecodes_holder);

  // Create buffer from typed pointer and lifetime capsule
  auto bytecodes_array = py::array_t<I>(bytecodes_holder->size(), bytecodes_holder->data(), bytecodes_capsule);
  if (index >= length || index < 0) {
    throw std::invalid_argument(
        std::string("out of bounds index in ForthMachineOf.__getitem__: ")
        + FILENAME(__LINE__));
  }
  // Build Python objects for buffer sublist
  auto start = offsets[index];
  auto stop = offsets[index+1];
  return bytecodes_array[py::slice(start, stop, 1)];


}


template <typename T, typename I>
py::class_<ak::ForthMachineOf<T, I>, std::shared_ptr<ak::ForthMachineOf<T, I>>>
make_ForthMachineOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::ForthMachineOf<T, I>,
          std::shared_ptr<ak::ForthMachineOf<T, I>>>(m, name.c_str())
          .def(py::init([](const std::string& source,
                           int64_t stack_size,
                           int64_t recursion_depth,
                           int64_t string_buffer_size,
                           int64_t output_initial_size,
                           double output_resize_factor)
                        -> std::shared_ptr<ak::ForthMachineOf<T, I>> {
            return std::make_shared<ak::ForthMachineOf<T, I>>(source,
                                                              stack_size,
                                                              recursion_depth,
                                                              string_buffer_size,
                                                              output_initial_size,
                                                              output_resize_factor);
          }),
               py::arg("source"),
               py::arg("stack_size") = 1024,
               py::arg("recursion_depth") = 1024,
               py::arg("string_buffer_size") = 1024,
               py::arg("output_initial_size") = 1024,
               py::arg("output_resize_factor") = 1.5)
          .def("__getitem__", [](const std::shared_ptr<ak::ForthMachineOf<T, I>>& self,
                                 const std::string& key)
                                 -> py::object {
            if (self.get()->is_variable(key)) {
              T out = self.get()->variable_at(key);
              return py::cast(out);
            }
            else if (self.get()->is_output(key)) {
                auto output = self.get()->output_at(key);
                return output_buffer_to_numpy(output);
            }
            else if (self.get()->is_defined(key)) {
              const std::vector<std::string> dictionary = self.get()->dictionary();
              int64_t index = 0;
              for (;  (size_t)index < dictionary.size();  index++) {
                if (dictionary[index] == key) {
                  break;
                }
              }
              return machine_bytecodes_at_to_python_content(self, index+1);
            }
            else {
                throw std::invalid_argument(
                    std::string("unrecognized AwkwardForth variable/output/dictionary word: ")
                    + key + FILENAME(__LINE__));
            }
          })
          .def_property_readonly("abi_version",
              &ak::ForthMachineOf<T, I>::abi_version)
          .def_property_readonly("source",
              &ak::ForthMachineOf<T, I>::source)
          .def_property_readonly("bytecodes",
              &ak::ForthMachineOf<T, I>::bytecodes)
          .def_property_readonly("bytecodes_offsets",
              &ak::ForthMachineOf<T, I>::bytecodes_offsets)
          .def_property_readonly("decompiled",
              &ak::ForthMachineOf<T, I>::decompiled)
          .def_property_readonly("dictionary",
              &ak::ForthMachineOf<T, I>::dictionary)
          .def_property_readonly("stack_max_depth",
              &ak::ForthMachineOf<T, I>::stack_max_depth)
          .def_property_readonly("recursion_max_depth",
              &ak::ForthMachineOf<T, I>::recursion_max_depth)
          .def_property_readonly("string_buffer_size",
              &ak::ForthMachineOf<T, I>::string_buffer_size)
          .def_property_readonly("output_initial_size",
              &ak::ForthMachineOf<T, I>::output_initial_size)
          .def_property_readonly("output_resize_factor",
              &ak::ForthMachineOf<T, I>::output_resize_factor)
          .def_property_readonly("stack",
              &ak::ForthMachineOf<T, I>::stack)
          .def("stack_push", [](ak::ForthMachineOf<T, I>& self, T value) -> void {
              if (!self.stack_can_push()) {
                throw std::invalid_argument(std::string("AwkwardForth stack overflow")
                                            + FILENAME(__LINE__));
              }
              self.stack_push(value);
          })
          .def("stack_pop", [](ak::ForthMachineOf<T, I>& self) -> T {
              if (!self.stack_can_pop()) {
                throw std::invalid_argument(std::string("AwkwardForth stack underflow")
                                            + FILENAME(__LINE__));
              }
              return self.stack_pop();
          })
          .def("stack_clear", &ak::ForthMachineOf<T, I>::stack_clear)
          .def("string_at", [](ak::ForthMachineOf<T, I>& self,
                               int64_t at) -> py::str {
            return self.string_at(at);
          })
          .def_property_readonly("variables", &ak::ForthMachineOf<T, I>::variables)
          .def("input_position", [](ak::ForthMachineOf<T, I>& self,
                                    const std::string& name) -> int64_t {
              return self.input_position_at(name);
          })
          .def_property_readonly("outputs",
            [](const std::shared_ptr<ak::ForthMachineOf<T, I>> self) -> py::dict {
              py::dict out;
              for (auto name : self.get()->output_index()) {
                py::object pyname = py::cast(name);
                auto output = self.get()->output_at(name);
                out[pyname] = output_buffer_to_numpy(output);
              }
              return out;
          })
          .def("output",
            [](const std::shared_ptr<ak::ForthMachineOf<T, I>> self,
               const std::string& name) -> py::object {
              auto output = self.get()->output_at(name);
              return output_buffer_to_numpy(output);
          })
          .def("reset", &ak::ForthMachineOf<T, I>::reset)
          .def("begin", [](ak::ForthMachineOf<T, I>& self,
                           const py::dict& inputs) -> void {
              std::map<std::string, std::shared_ptr<ak::ForthInputBuffer>> ins;
              for (auto pair : inputs) {
                std::string name = pair.first.cast<std::string>();
                py::buffer obj = pair.second.cast<py::buffer>();
                py::buffer_info info = obj.request(self.input_must_be_writable(name));
                int64_t length = info.itemsize;
                for (auto x : info.shape) {
                  length *= x;
                }
                std::shared_ptr<void> ptr = std::shared_ptr<uint8_t>(
                    reinterpret_cast<uint8_t*>(info.ptr), pyobject_deleter<uint8_t>(obj.ptr()));
                ins[name] = std::make_shared<ak::ForthInputBuffer>(ptr, 0, length);
              }
              self.begin(ins);
          }, py::arg("inputs") = py::dict())
          .def("begin_again", [](ak::ForthMachineOf<T, I>& self,
                           const py::dict& inputs, bool reset_instruction) -> void {
              std::map<std::string, std::shared_ptr<ak::ForthInputBuffer>> ins;
              for (auto pair : inputs) {
                std::string name = pair.first.cast<std::string>();
                py::buffer obj = pair.second.cast<py::buffer>();
                py::buffer_info info = obj.request(self.input_must_be_writable(name));
                int64_t length = info.itemsize;
                for (auto x : info.shape) {
                  length *= x;
                }
                std::shared_ptr<void> ptr = std::shared_ptr<uint8_t>(
                    reinterpret_cast<uint8_t*>(info.ptr), pyobject_deleter<uint8_t>(obj.ptr()));
                ins[name] = std::make_shared<ak::ForthInputBuffer>(ptr, 0, length);
              }
              self.begin_again(ins, reset_instruction);
          }, py::arg("inputs"), py::arg("reset_instruction") = true)
          .def("step", [](ak::ForthMachineOf<T, I>& self,
                          bool raise_user_halt,
                          bool raise_recursion_depth_exceeded,
                          bool raise_stack_underflow,
                          bool raise_stack_overflow,
                          bool raise_read_beyond,
                          bool raise_seek_beyond,
                          bool raise_skip_beyond,
                          bool raise_rewind_beyond,
                          bool raise_division_by_zero,
                          bool raise_varint_too_big,
                          bool raise_text_number_missing,
                          bool raise_quoted_string_missing,
                          bool raise_enumeration_missing) -> py::object {
              ak::util::ForthError err = self.step();
              return maybe_throw<T, I>(self,
                                       err,
                                       raise_user_halt,
                                       raise_recursion_depth_exceeded,
                                       raise_stack_underflow,
                                       raise_stack_overflow,
                                       raise_read_beyond,
                                       raise_seek_beyond,
                                       raise_skip_beyond,
                                       raise_rewind_beyond,
                                       raise_division_by_zero,
                                       raise_varint_too_big,
                                       raise_text_number_missing,
                                       raise_quoted_string_missing,
                                       raise_enumeration_missing);
          }, py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true
           , py::arg("raise_division_by_zero") = true
           , py::arg("raise_varint_too_big") = true
           , py::arg("raise_text_number_missing") = true
           , py::arg("raise_quoted_string_missing") = true
           , py::arg("raise_enumeration_missing") = true)
          .def("run", [](ak::ForthMachineOf<T, I>& self,
                         const py::dict& inputs,
                         bool raise_user_halt,
                         bool raise_recursion_depth_exceeded,
                         bool raise_stack_underflow,
                         bool raise_stack_overflow,
                         bool raise_read_beyond,
                         bool raise_seek_beyond,
                         bool raise_skip_beyond,
                         bool raise_rewind_beyond,
                         bool raise_division_by_zero,
                         bool raise_varint_too_big,
                         bool raise_text_number_missing,
                         bool raise_quoted_string_missing,
                         bool raise_enumeration_missing) -> py::object {
              std::map<std::string, std::shared_ptr<ak::ForthInputBuffer>> ins;
              for (auto pair : inputs) {
                std::string name = pair.first.cast<std::string>();
                py::buffer obj = pair.second.cast<py::buffer>();
                py::buffer_info info = obj.request(self.input_must_be_writable(name));
                int64_t length = info.itemsize;
                for (auto x : info.shape) {
                  length *= x;
                }
                std::shared_ptr<void> ptr = std::shared_ptr<uint8_t>(
                    reinterpret_cast<uint8_t*>(info.ptr), pyobject_deleter<uint8_t>(obj.ptr()));
                ins[name] = std::make_shared<ak::ForthInputBuffer>(ptr, 0, length);
              }
              self.begin(ins);
              py::gil_scoped_release release;
              ak::util::ForthError err = self.resume();
              py::gil_scoped_acquire acquire;
              return maybe_throw<T, I>(self,
                                       err,
                                       raise_user_halt,
                                       raise_recursion_depth_exceeded,
                                       raise_stack_underflow,
                                       raise_stack_overflow,
                                       raise_read_beyond,
                                       raise_seek_beyond,
                                       raise_skip_beyond,
                                       raise_rewind_beyond,
                                       raise_division_by_zero,
                                       raise_varint_too_big,
                                       raise_text_number_missing,
                                       raise_quoted_string_missing,
                                       raise_enumeration_missing);
          }, py::arg("inputs") = py::dict()
           , py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true
           , py::arg("raise_division_by_zero") = true
           , py::arg("raise_varint_too_big") = true
           , py::arg("raise_text_number_missing") = true
           , py::arg("raise_quoted_string_missing") = true
           , py::arg("raise_enumeration_missing") = true)
          .def("resume", [](ak::ForthMachineOf<T, I>& self,
                          bool raise_user_halt,
                          bool raise_recursion_depth_exceeded,
                          bool raise_stack_underflow,
                          bool raise_stack_overflow,
                          bool raise_read_beyond,
                          bool raise_seek_beyond,
                          bool raise_skip_beyond,
                          bool raise_rewind_beyond,
                          bool raise_division_by_zero,
                          bool raise_varint_too_big,
                          bool raise_text_number_missing,
                          bool raise_quoted_string_missing,
                          bool raise_enumeration_missing) -> py::object {
              py::gil_scoped_release release;
              ak::util::ForthError err = self.resume();
              py::gil_scoped_acquire acquire;
              return maybe_throw<T, I>(self,
                                       err,
                                       raise_user_halt,
                                       raise_recursion_depth_exceeded,
                                       raise_stack_underflow,
                                       raise_stack_overflow,
                                       raise_read_beyond,
                                       raise_seek_beyond,
                                       raise_skip_beyond,
                                       raise_rewind_beyond,
                                       raise_division_by_zero,
                                       raise_varint_too_big,
                                       raise_text_number_missing,
                                       raise_quoted_string_missing,
                                       raise_enumeration_missing);
          }, py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true
           , py::arg("raise_division_by_zero") = true
           , py::arg("raise_varint_too_big") = true
           , py::arg("raise_text_number_missing") = true
           , py::arg("raise_quoted_string_missing") = true
           , py::arg("raise_enumeration_missing") = true)
          .def("call", [](ak::ForthMachineOf<T, I>& self,
                          const std::string& name,
                          bool raise_user_halt,
                          bool raise_recursion_depth_exceeded,
                          bool raise_stack_underflow,
                          bool raise_stack_overflow,
                          bool raise_read_beyond,
                          bool raise_seek_beyond,
                          bool raise_skip_beyond,
                          bool raise_rewind_beyond,
                          bool raise_division_by_zero,
                          bool raise_varint_too_big,
                          bool raise_text_number_missing,
                          bool raise_quoted_string_missing,
                          bool raise_enumeration_missing) -> py::object {
              py::gil_scoped_release release;
              ak::util::ForthError err = self.call(name);
              py::gil_scoped_acquire acquire;
              return maybe_throw<T, I>(self,
                                       err,
                                       raise_user_halt,
                                       raise_recursion_depth_exceeded,
                                       raise_stack_underflow,
                                       raise_stack_overflow,
                                       raise_read_beyond,
                                       raise_seek_beyond,
                                       raise_skip_beyond,
                                       raise_rewind_beyond,
                                       raise_division_by_zero,
                                       raise_varint_too_big,
                                       raise_text_number_missing,
                                       raise_quoted_string_missing,
                                       raise_enumeration_missing);
          }, py::arg("name")
           , py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true
           , py::arg("raise_division_by_zero") = true
           , py::arg("raise_varint_too_big") = true
           , py::arg("raise_text_number_missing") = true
           , py::arg("raise_quoted_string_missing") = true
           , py::arg("raise_enumeration_missing") = true)
          .def_property_readonly("current_bytecode_position",
              &ak::ForthMachineOf<T, I>::current_bytecode_position)
          .def_property_readonly("current_recursion_depth",
              &ak::ForthMachineOf<T, I>::current_recursion_depth)
          .def_property_readonly("current_instruction",
              &ak::ForthMachineOf<T, I>::current_instruction)
          .def("count_reset",
              &ak::ForthMachineOf<T, I>::count_reset)
          .def_property_readonly("count_instructions",
              &ak::ForthMachineOf<T, I>::count_instructions)
          .def_property_readonly("count_reads",
              &ak::ForthMachineOf<T, I>::count_reads)
          .def_property_readonly("count_writes",
              &ak::ForthMachineOf<T, I>::count_writes)
          .def_property_readonly("count_nanoseconds",
              &ak::ForthMachineOf<T, I>::count_nanoseconds)
          .def("is_variable",
              &ak::ForthMachineOf<T, I>::is_variable)
          .def("is_input",
              &ak::ForthMachineOf<T, I>::is_input)
          .def("is_output",
              &ak::ForthMachineOf<T, I>::is_output)
          .def("is_defined",
              &ak::ForthMachineOf<T, I>::is_defined)
          .def_property_readonly("is_ready",
              &ak::ForthMachineOf<T, I>::is_ready)
          .def_property_readonly("is_done",
              &ak::ForthMachineOf<T, I>::is_done)
          .def_property_readonly("is_segment_done",
              &ak::ForthMachineOf<T, I>::is_segment_done)

         );
}

template py::class_<ak::ForthMachine32, std::shared_ptr<ak::ForthMachine32>>
make_ForthMachineOf(const py::handle& m, const std::string& name);

template py::class_<ak::ForthMachine64, std::shared_ptr<ak::ForthMachine64>>
make_ForthMachineOf(const py::handle& m, const std::string& name);

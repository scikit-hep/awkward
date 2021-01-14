// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/forth.cpp", line)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "awkward/python/util.h"
#include "awkward/python/content.h"
#include "awkward/python/forth.h"

py::array
output_to_python(const py::object& self,
                 const std::shared_ptr<ak::ForthOutputBuffer>& buff) {
  void* ptr = buff.get()->ptr().get();
  int64_t length = buff.get()->len();
  if (dynamic_cast<ak::ForthOutputBufferOf<bool>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(bool),
                                     py::format_descriptor<bool>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(int64_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<int8_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(int8_t),
                                     py::format_descriptor<int8_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(int8_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<int16_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(int16_t),
                                     py::format_descriptor<int16_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(int16_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<int32_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(int32_t),
                                     py::format_descriptor<int32_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(int32_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<int64_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(int64_t),
                                     py::format_descriptor<int64_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(int64_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<uint8_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(uint8_t),
                                     py::format_descriptor<uint8_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(uint8_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<uint16_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(uint16_t),
                                     py::format_descriptor<uint16_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(uint16_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<uint32_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(uint32_t),
                                     py::format_descriptor<uint32_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(uint32_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<uint64_t>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(uint64_t),
                                     py::format_descriptor<uint64_t>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(uint64_t) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<float>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(float),
                                     py::format_descriptor<float>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(float) }),
                     self);
  }
  else if (dynamic_cast<ak::ForthOutputBufferOf<double>*>(buff.get())) {
    return py::array(py::buffer_info(ptr,
                                     sizeof(double),
                                     py::format_descriptor<double>::format(),
                                     1,
                                     { (ssize_t)length },
                                     { (ssize_t)sizeof(double) }),
                     self);
  }
  else {
    throw std::runtime_error(
        std::string("unrecognized ForthOutputBuffer specialization")
        + FILENAME(__LINE__));
  }
}

template <typename T, typename I>
py::class_<ak::ForthMachineOf<T, I>, std::shared_ptr<ak::ForthMachineOf<T, I>>>
make_ForthMachineOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::ForthMachineOf<T, I>,
          std::shared_ptr<ak::ForthMachineOf<T, I>>>(m, name.c_str())
          .def(py::init([](const std::string& source,
                           int64_t stack_size,
                           int64_t recursion_depth,
                           int64_t output_initial_size,
                           double output_resize_factor)
                        -> std::shared_ptr<ak::ForthMachineOf<T, I>> {
            return std::make_shared<ak::ForthMachineOf<T, I>>(source,
                                                              stack_size,
                                                              recursion_depth,
                                                              output_initial_size,
                                                              output_resize_factor);
          }),
               py::arg("source"),
               py::arg("stack_size") = 1024,
               py::arg("recursion_depth") = 1024,
               py::arg("output_initial_size") = 1024,
               py::arg("output_resize_factor") = 1.5)
          .def("__getitem__", [](const py::object& self, const std::string& key)
                                 -> py::object {
            const std::shared_ptr<ak::ForthMachineOf<T, I>> myself =
                self.cast<std::shared_ptr<ak::ForthMachineOf<T, I>>>();
            if (myself.get()->is_variable(key)) {
              T out = myself.get()->variable_at(key);
              return py::cast(out);
            }
            else if (myself.get()->is_output(key)) {
              const std::shared_ptr<ak::ForthOutputBuffer>& buff =
                  myself.get()->output_at(key);
              return output_to_python(self, buff);
            }
            else if (myself.get()->is_defined(key)) {
              const std::vector<std::string> dictionary = myself.get()->dictionary();
              int64_t index = 0;
              for (;  index < dictionary.size();  index++) {
                if (dictionary[index] == key) {
                  break;
                }
              }
              ak::ContentPtr bytecodes = myself.get()->bytecodes();
              ak::ContentPtr out = bytecodes.get()->getitem_at_nowrap(index + 1);
              return box(out);
            }
            else {
                throw std::invalid_argument(
                    std::string("unrecognized Awkward Forth variable/output/dictionary word: ")
                    + key + FILENAME(__LINE__));
            }
          })
          .def_property_readonly("source",
              &ak::ForthMachineOf<T, I>::source)
          .def_property_readonly("bytecodes",
              &ak::ForthMachineOf<T, I>::bytecodes)
          .def_property_readonly("assembly_instructions",
              &ak::ForthMachineOf<T, I>::assembly_instructions)
          .def_property_readonly("dictionary",
              &ak::ForthMachineOf<T, I>::dictionary)
          .def_property_readonly("stack_max_depth",
              &ak::ForthMachineOf<T, I>::stack_max_depth)
          .def_property_readonly("recursion_max_depth",
              &ak::ForthMachineOf<T, I>::recursion_max_depth)
          .def_property_readonly("output_initial_size",
              &ak::ForthMachineOf<T, I>::output_initial_size)
          .def_property_readonly("output_resize_factor",
              &ak::ForthMachineOf<T, I>::output_resize_factor)
          .def_property_readonly("stack",
              &ak::ForthMachineOf<T, I>::stack)
          .def("stack_push", [](ak::ForthMachineOf<T, I>& self, T value) -> void {
              if (!self.stack_can_push()) {
                throw std::invalid_argument(std::string("Awkward Forth stack overflow")
                                            + FILENAME(__LINE__));
              }
              self.stack_push(value);
          })
          .def("stack_pop", [](ak::ForthMachineOf<T, I>& self) -> T {
              if (!self.stack_can_pop()) {
                throw std::invalid_argument(std::string("Awkward Forth stack underflow")
                                            + FILENAME(__LINE__));
              }
              return self.stack_pop();
          })
          .def("stack_clear", &ak::ForthMachineOf<T, I>::stack_clear)
          .def_property_readonly("variables", &ak::ForthMachineOf<T, I>::variables)
          .def_property_readonly("outputs", [](const py::object& self) -> py::dict {
              const std::shared_ptr<ak::ForthMachineOf<T, I>> myself =
                self.cast<std::shared_ptr<ak::ForthMachineOf<T, I>>>();
              py::dict out;
              for (auto name : myself.get()->output_index()) {
                py::object pyname = py::cast(name);
                const std::shared_ptr<ak::ForthOutputBuffer>& buff =
                    myself.get()->output_at(name);
                out[pyname] = output_to_python(self, buff);
              }
              return out;
          })
          .def("reset", &ak::ForthMachineOf<T, I>::reset)
          .def("begin", [](ak::ForthMachineOf<T, I>& self,
                           const py::dict& inputs) -> void {
              std::map<std::string, std::shared_ptr<ak::ForthInputBuffer>> ins;
              for (auto pair : inputs) {
                std::string name = pair.first.cast<std::string>();
                py::buffer obj = pair.second.cast<py::buffer>();
                py::buffer_info info = obj.request(true);
                int64_t length = 1;
                for (auto x : info.shape) {
                  length *= x;
                }
                std::shared_ptr<void> ptr = std::shared_ptr<uint8_t>(
                    reinterpret_cast<uint8_t*>(info.ptr), pyobject_deleter<uint8_t>(obj.ptr()));
                ins[name] = std::make_shared<ak::ForthInputBuffer>(ptr, 0, length);
              }
              self.begin(ins);
          }, py::arg("inputs") = py::dict())
          .def("step", [](ak::ForthMachineOf<T, I>& self,
                          bool ignore_recursion_depth_exceeded,
                          bool ignore_stack_underflow,
                          bool ignore_stack_overflow,
                          bool ignore_read_beyond,
                          bool ignore_seek_beyond,
                          bool ignore_skip_beyond,
                          bool ignore_rewind_beyond) -> void {
              if (!self.is_ready()) {
                throw std::invalid_argument(
                   std::string("Awkward Forth machine is not ready; call 'begin' first")
                   + FILENAME(__LINE__));
              }
              else if (self.is_done()) {
                throw std::invalid_argument(
                   std::string("Awkward Forth machine has reached the end of its program")
                   + FILENAME(__LINE__));
              }
              else {
                ak::util::ForthError err = self.step();
                std::set<ak::util::ForthError> ignore;
                if (ignore_recursion_depth_exceeded) {
                  ignore.insert(ak::util::ForthError::recursion_depth_exceeded);
                }
                if (ignore_stack_underflow) {
                  ignore.insert(ak::util::ForthError::stack_underflow);
                }
                if (ignore_stack_overflow) {
                  ignore.insert(ak::util::ForthError::stack_overflow);
                }
                if (ignore_read_beyond) {
                  ignore.insert(ak::util::ForthError::read_beyond);
                }
                if (ignore_seek_beyond) {
                  ignore.insert(ak::util::ForthError::seek_beyond);
                }
                if (ignore_skip_beyond) {
                  ignore.insert(ak::util::ForthError::skip_beyond);
                }
                if (ignore_rewind_beyond) {
                  ignore.insert(ak::util::ForthError::rewind_beyond);
                }
                self.maybe_throw(err, ignore);
              }

          }, py::arg("ignore_recursion_depth_exceeded") = false
           , py::arg("ignore_stack_underflow") = false
           , py::arg("ignore_stack_overflow") = false
           , py::arg("ignore_read_beyond") = false
           , py::arg("ignore_seek_beyond") = false
           , py::arg("ignore_skip_beyond") = false
           , py::arg("ignore_rewind_beyond") = false)
          .def("run", [](ak::ForthMachineOf<T, I>& self,
                         const py::dict& inputs,
                         bool ignore_recursion_depth_exceeded,
                         bool ignore_stack_underflow,
                         bool ignore_stack_overflow,
                         bool ignore_read_beyond,
                         bool ignore_seek_beyond,
                         bool ignore_skip_beyond,
                         bool ignore_rewind_beyond) -> void {
              std::map<std::string, std::shared_ptr<ak::ForthInputBuffer>> ins;
              for (auto pair : inputs) {
                std::string name = pair.first.cast<std::string>();
                py::buffer obj = pair.second.cast<py::buffer>();
                py::buffer_info info = obj.request(true);
                int64_t length = 1;
                for (auto x : info.shape) {
                  length *= x;
                }
                std::shared_ptr<void> ptr = std::shared_ptr<uint8_t>(
                    reinterpret_cast<uint8_t*>(info.ptr), pyobject_deleter<uint8_t>(obj.ptr()));
                ins[name] = std::make_shared<ak::ForthInputBuffer>(ptr, 0, length);
              }
              ak::util::ForthError err = self.run(ins);
              std::set<ak::util::ForthError> ignore;
              if (ignore_recursion_depth_exceeded) {
                ignore.insert(ak::util::ForthError::recursion_depth_exceeded);
              }
              if (ignore_stack_underflow) {
                ignore.insert(ak::util::ForthError::stack_underflow);
              }
              if (ignore_stack_overflow) {
                ignore.insert(ak::util::ForthError::stack_overflow);
              }
              if (ignore_read_beyond) {
                ignore.insert(ak::util::ForthError::read_beyond);
              }
              if (ignore_seek_beyond) {
                ignore.insert(ak::util::ForthError::seek_beyond);
              }
              if (ignore_skip_beyond) {
                ignore.insert(ak::util::ForthError::skip_beyond);
              }
              if (ignore_rewind_beyond) {
                ignore.insert(ak::util::ForthError::rewind_beyond);
              }
              self.maybe_throw(err, ignore);
          }, py::arg("inputs") = py::dict()
           , py::arg("ignore_recursion_depth_exceeded") = false
           , py::arg("ignore_stack_underflow") = false
           , py::arg("ignore_stack_overflow") = false
           , py::arg("ignore_read_beyond") = false
           , py::arg("ignore_seek_beyond") = false
           , py::arg("ignore_skip_beyond") = false
           , py::arg("ignore_rewind_beyond") = false)
          .def("resume", [](ak::ForthMachineOf<T, I>& self,
                          bool ignore_recursion_depth_exceeded,
                          bool ignore_stack_underflow,
                          bool ignore_stack_overflow,
                          bool ignore_read_beyond,
                          bool ignore_seek_beyond,
                          bool ignore_skip_beyond,
                          bool ignore_rewind_beyond) -> void {
              if (!self.is_ready()) {
                throw std::invalid_argument(
                   std::string("Awkward Forth machine is not ready; call 'begin' first")
                   + FILENAME(__LINE__));
              }
              else if (self.is_done()) {
                throw std::invalid_argument(
                   std::string("Awkward Forth machine has reached the end of its program")
                   + FILENAME(__LINE__));
              }
              else {
                ak::util::ForthError err = self.resume();
                std::set<ak::util::ForthError> ignore;
                if (ignore_recursion_depth_exceeded) {
                  ignore.insert(ak::util::ForthError::recursion_depth_exceeded);
                }
                if (ignore_stack_underflow) {
                  ignore.insert(ak::util::ForthError::stack_underflow);
                }
                if (ignore_stack_overflow) {
                  ignore.insert(ak::util::ForthError::stack_overflow);
                }
                if (ignore_read_beyond) {
                  ignore.insert(ak::util::ForthError::read_beyond);
                }
                if (ignore_seek_beyond) {
                  ignore.insert(ak::util::ForthError::seek_beyond);
                }
                if (ignore_skip_beyond) {
                  ignore.insert(ak::util::ForthError::skip_beyond);
                }
                if (ignore_rewind_beyond) {
                  ignore.insert(ak::util::ForthError::rewind_beyond);
                }
                self.maybe_throw(err, ignore);
              }
          }, py::arg("ignore_recursion_depth_exceeded") = false
           , py::arg("ignore_stack_underflow") = false
           , py::arg("ignore_stack_overflow") = false
           , py::arg("ignore_read_beyond") = false
           , py::arg("ignore_seek_beyond") = false
           , py::arg("ignore_skip_beyond") = false
           , py::arg("ignore_rewind_beyond") = false)
          .def("call", [](ak::ForthMachineOf<T, I>& self,
                          const std::string& name,
                          bool ignore_recursion_depth_exceeded,
                          bool ignore_stack_underflow,
                          bool ignore_stack_overflow,
                          bool ignore_read_beyond,
                          bool ignore_seek_beyond,
                          bool ignore_skip_beyond,
                          bool ignore_rewind_beyond) -> void {
              if (!self.is_ready()) {
                throw std::invalid_argument(
                   std::string("Awkward Forth machine is not ready; call 'begin' first")
                   + FILENAME(__LINE__));
              }
              else {
                ak::util::ForthError err = self.call(name);
                std::set<ak::util::ForthError> ignore;
                if (ignore_recursion_depth_exceeded) {
                  ignore.insert(ak::util::ForthError::recursion_depth_exceeded);
                }
                if (ignore_stack_underflow) {
                  ignore.insert(ak::util::ForthError::stack_underflow);
                }
                if (ignore_stack_overflow) {
                  ignore.insert(ak::util::ForthError::stack_overflow);
                }
                if (ignore_read_beyond) {
                  ignore.insert(ak::util::ForthError::read_beyond);
                }
                if (ignore_seek_beyond) {
                  ignore.insert(ak::util::ForthError::seek_beyond);
                }
                if (ignore_skip_beyond) {
                  ignore.insert(ak::util::ForthError::skip_beyond);
                }
                if (ignore_rewind_beyond) {
                  ignore.insert(ak::util::ForthError::rewind_beyond);
                }
                self.maybe_throw(err, ignore);
              }
          }, py::arg("name")
           , py::arg("ignore_recursion_depth_exceeded") = false
           , py::arg("ignore_stack_underflow") = false
           , py::arg("ignore_stack_overflow") = false
           , py::arg("ignore_read_beyond") = false
           , py::arg("ignore_seek_beyond") = false
           , py::arg("ignore_skip_beyond") = false
           , py::arg("ignore_rewind_beyond") = false)
          .def_property_readonly("breakpoint_depth",
              &ak::ForthMachineOf<T, I>::breakpoint_depth)
          .def_property_readonly("current_bytecode",
              &ak::ForthMachineOf<T, I>::current_bytecode)
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

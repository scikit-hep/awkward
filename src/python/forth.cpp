// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/forth.cpp", line)

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
                       bool raise_rewind_beyond) {
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
  self.maybe_throw(err, ignore);

  switch (err) {
    case ak::util::ForthError::none:
      return py::none();
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
    default:
      throw std::invalid_argument(
          std::string("unrecognized ForthError: ")
          + std::to_string((int64_t)err) + FILENAME(__LINE__));
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
          .def("__getitem__", [](const std::shared_ptr<ak::ForthMachineOf<T, I>>& self,
                                 const std::string& key)
                                 -> py::object {
            if (self.get()->is_variable(key)) {
              T out = self.get()->variable_at(key);
              return py::cast(out);
            }
            else if (self.get()->is_output(key)) {
              return box(self.get()->output_NumpyArray_at(key));
            }
            else if (self.get()->is_defined(key)) {
              const std::vector<std::string> dictionary = self.get()->dictionary();
              int64_t index = 0;
              for (;  index < dictionary.size();  index++) {
                if (dictionary[index] == key) {
                  break;
                }
              }
              ak::ContentPtr bytecodes = self.get()->bytecodes();
              return box(bytecodes.get()->getitem_at_nowrap(index + 1));
            }
            else {
                throw std::invalid_argument(
                    std::string("unrecognized AwkwardForth variable/output/dictionary word: ")
                    + key + FILENAME(__LINE__));
            }
          })
          .def_property_readonly("source",
              &ak::ForthMachineOf<T, I>::source)
          .def_property_readonly("bytecodes",
              &ak::ForthMachineOf<T, I>::bytecodes)
          .def_property_readonly("decompiled",
              &ak::ForthMachineOf<T, I>::decompiled)
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
          .def_property_readonly("variables", &ak::ForthMachineOf<T, I>::variables)
          .def_property_readonly("outputs",
            [](const std::shared_ptr<ak::ForthMachineOf<T, I>> self) -> py::dict {
              py::dict out;
              for (auto name : self.get()->output_index()) {
                py::object pyname = py::cast(name);
                out[pyname] = box(self.get()->output_NumpyArray_at(name));
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
                          bool raise_user_halt,
                          bool raise_recursion_depth_exceeded,
                          bool raise_stack_underflow,
                          bool raise_stack_overflow,
                          bool raise_read_beyond,
                          bool raise_seek_beyond,
                          bool raise_skip_beyond,
                          bool raise_rewind_beyond) -> void {
              if (!self.is_ready()) {
                throw std::invalid_argument(
                   std::string("AwkwardForth machine is not ready; call 'begin' first")
                   + FILENAME(__LINE__));
              }
              else if (self.is_done()) {
                throw std::invalid_argument(
                   std::string("AwkwardForth machine is at the end of its program; cannot 'step' again")
                   + FILENAME(__LINE__));
              }
              else {
                ak::util::ForthError err = self.step();
                maybe_throw<T, I>(self,
                                  err,
                                  raise_user_halt,
                                  raise_recursion_depth_exceeded,
                                  raise_stack_underflow,
                                  raise_stack_overflow,
                                  raise_read_beyond,
                                  raise_seek_beyond,
                                  raise_skip_beyond,
                                  raise_rewind_beyond);
              }

          }, py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true)
          .def("run", [](ak::ForthMachineOf<T, I>& self,
                         const py::dict& inputs,
                         bool raise_user_halt,
                         bool raise_recursion_depth_exceeded,
                         bool raise_stack_underflow,
                         bool raise_stack_overflow,
                         bool raise_read_beyond,
                         bool raise_seek_beyond,
                         bool raise_skip_beyond,
                         bool raise_rewind_beyond) -> void {
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
              maybe_throw<T, I>(self,
                                err,
                                raise_user_halt,
                                raise_recursion_depth_exceeded,
                                raise_stack_underflow,
                                raise_stack_overflow,
                                raise_read_beyond,
                                raise_seek_beyond,
                                raise_skip_beyond,
                                raise_rewind_beyond);
          }, py::arg("inputs") = py::dict()
           , py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true)
          .def("resume", [](ak::ForthMachineOf<T, I>& self,
                          bool raise_user_halt,
                          bool raise_recursion_depth_exceeded,
                          bool raise_stack_underflow,
                          bool raise_stack_overflow,
                          bool raise_read_beyond,
                          bool raise_seek_beyond,
                          bool raise_skip_beyond,
                          bool raise_rewind_beyond) -> void {
              if (!self.is_ready()) {
                throw std::invalid_argument(
                   std::string("AwkwardForth machine is not ready; call 'begin' first")
                   + FILENAME(__LINE__));
              }
              else if (self.is_done()) {
                throw std::invalid_argument(
                   std::string("AwkwardForth machine has reached the end of its program")
                   + FILENAME(__LINE__));
              }
              else {
                ak::util::ForthError err = self.resume();
                maybe_throw<T, I>(self,
                                  err,
                                  raise_user_halt,
                                  raise_recursion_depth_exceeded,
                                  raise_stack_underflow,
                                  raise_stack_overflow,
                                  raise_read_beyond,
                                  raise_seek_beyond,
                                  raise_skip_beyond,
                                  raise_rewind_beyond);
              }
          }, py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true)
          .def("call", [](ak::ForthMachineOf<T, I>& self,
                          const std::string& name,
                          bool raise_user_halt,
                          bool raise_recursion_depth_exceeded,
                          bool raise_stack_underflow,
                          bool raise_stack_overflow,
                          bool raise_read_beyond,
                          bool raise_seek_beyond,
                          bool raise_skip_beyond,
                          bool raise_rewind_beyond) -> void {
              if (!self.is_ready()) {
                throw std::invalid_argument(
                   std::string("AwkwardForth machine is not ready; call 'begin' first")
                   + FILENAME(__LINE__));
              }
              else {
                ak::util::ForthError err = self.call(name);
                maybe_throw<T, I>(self,
                                  err,
                                  raise_user_halt,
                                  raise_recursion_depth_exceeded,
                                  raise_stack_underflow,
                                  raise_stack_overflow,
                                  raise_read_beyond,
                                  raise_seek_beyond,
                                  raise_skip_beyond,
                                  raise_rewind_beyond);

              }
          }, py::arg("name")
           , py::arg("raise_user_halt") = true
           , py::arg("raise_recursion_depth_exceeded") = true
           , py::arg("raise_stack_underflow") = true
           , py::arg("raise_stack_overflow") = true
           , py::arg("raise_read_beyond") = true
           , py::arg("raise_seek_beyond") = true
           , py::arg("raise_skip_beyond") = true
           , py::arg("raise_rewind_beyond") = true)
          .def_property_readonly("pause_depth",
              &ak::ForthMachineOf<T, I>::pause_depth)
          .def_property_readonly("current_bytecode",
              &ak::ForthMachineOf<T, I>::current_bytecode)
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

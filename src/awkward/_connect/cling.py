# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import ctypes
import json
import re

from awkward_cpp import libawkward

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


cache = {}


def generate_headers(compiler, use_cached=True):
    key = "headers"
    if use_cached:
        out = cache.get(key)
    else:
        out = None

    if out is None:
        out = """
#include <sys/types.h>
#include <vector>
#include <string>
#include <optional>  // C++17
#include <variant>  // C++17
#include <complex>
#include <chrono>

#include <Python.h>

extern "C" int printf(const char*, ...);
""".strip()
        cache[key] = out
        compiler(out)

    return out


def generate_ArrayView(compiler, use_cached=True):
    key = "ArrayView"
    if use_cached:
        out = cache.get(key)
    else:
        out = None

    if not use_cached or "headers" not in cache:
        generate_headers(compiler, use_cached=use_cached)

    if out is None:
        out = """
namespace awkward {
  template <typename ARRAY, typename VALUE>
  class Iterator {
  public:
    Iterator(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs), lookup_(lookup) { }

    VALUE operator*() const noexcept {
      return ARRAY(start_, stop_, which_, ptrs_, lookup_)[0];
    }

    void operator++() noexcept {
      start_++;
    }

    bool operator==(Iterator<ARRAY, VALUE> other) const noexcept {
      return start_ == other.start_   &&
             stop_ == other.stop_   &&
             which_ == other.which_   &&
             ptrs_ == other.ptrs_;
    }

    bool operator!=(Iterator<ARRAY, VALUE> other) const noexcept {
      return start_ != other.start_   ||
             stop_ != other.stop_   ||
             which_ != other.which_   ||
             ptrs_ != other.ptrs_;
    }

  private:
    ssize_t start_;
    ssize_t stop_;
    ssize_t which_;
    ssize_t* ptrs_;
    PyObject* lookup_;
  };

  template <typename ARRAY, typename VALUE>
  class RIterator {
  public:
    RIterator(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs), lookup_(lookup) { }

    VALUE operator*() const noexcept {
      return ARRAY(start_, stop_, which_, ptrs_, lookup_)[0];
    }

    void operator++() noexcept {
      start_--;
    }

    bool operator==(RIterator<ARRAY, VALUE> other) const noexcept {
      return start_ == other.start_   &&
             stop_ == other.stop_   &&
             which_ == other.which_   &&
             ptrs_ == other.ptrs_;
    }

    bool operator!=(RIterator<ARRAY, VALUE> other) const noexcept {
      return start_ != other.start_   ||
             stop_ != other.stop_   ||
             which_ != other.which_   ||
             ptrs_ != other.ptrs_;
    }

  private:
    ssize_t start_;
    ssize_t stop_;
    ssize_t which_;
    ssize_t* ptrs_;
    PyObject* lookup_;
  };

  class ArrayView {
  public:
    ArrayView(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs), lookup_(lookup) {
      }

    size_t size() const noexcept {
      return stop_ - start_;
    }

    bool empty() const noexcept {
      return start_ == stop_;
    }

    PyObject* lookup() {
        Py_INCREF(lookup_);
        return lookup_;
    }

  protected:
    // ROOT streamer customization is done by giving specific instructions
    // in the comments written after the declaration of data members: the values
    // of `ptrs_` and `lookup_` will be ignored (//!).
    // https://github.com/root-project/root/blob/master/io/doc/TFile/README.md#streamerinfo
    ssize_t start_;
    ssize_t stop_;
    ssize_t which_;
    ssize_t* ptrs_;    //! transient data pointer
    PyObject* lookup_; //! transient lookup
  };
}
""".strip()
        cache[key] = out
        compiler(out)

    return out


def generate_RecordView(compiler, use_cached=True):
    key = "RecordView"
    if use_cached:
        out = cache.get(key)
    else:
        out = None

    if not use_cached or "headers" not in cache:
        generate_headers(compiler, use_cached=use_cached)

    if out is None:
        out = """
namespace awkward {
  class RecordView {
  public:
    RecordView(ssize_t at, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : at_(at), which_(which), ptrs_(ptrs), lookup_(lookup) { }

    PyObject* lookup() {
        Py_INCREF(lookup_);
        return lookup_;
    }

  protected:
    // ROOT streamer customization is done by giving specific instructions
    // in the comments written after the declaration of data members: the values
    // of `ptrs_` and `lookup_` will be ignored (//!).
    // https://github.com/root-project/root/blob/master/io/doc/TFile/README.md#streamerinfo
    ssize_t at_;
    ssize_t which_;
    ssize_t* ptrs_;    //! transient data pointer
    PyObject* lookup_; //! transient lookup

  };
}
""".strip()
        cache[key] = out
        compiler(out)

    return out


def generate_ArrayBuilder(compiler, use_cached=True):
    key = "ArrayBuilder"
    if use_cached:
        out = cache.get(key)
    else:
        out = None

    if not use_cached or "headers" not in cache:
        generate_headers(compiler, use_cached=use_cached)

    if out is None:
        out = f"""
namespace awkward {{
  typedef unsigned char ArrayBuilderError;
  const ArrayBuilderError SUCCESS = 0;
  const ArrayBuilderError FAILURE = 1;

  typedef ArrayBuilderError (*ArrayBuilderMethod_length)(void*, int64_t*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_clear)(void*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_null)(void*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_boolean)(void*, bool);
  typedef ArrayBuilderError (*ArrayBuilderMethod_integer)(void*, int64_t);
  typedef ArrayBuilderError (*ArrayBuilderMethod_real)(void*, double);
  typedef ArrayBuilderError (*ArrayBuilderMethod_complex)(void*, double, double);
  typedef ArrayBuilderError (*ArrayBuilderMethod_datetime)(void*, int64_t, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_timedelta)(void*, int64_t, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_bytestring)(void*, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_bytestring_length)(void*, const char*, int64_t);
  typedef ArrayBuilderError (*ArrayBuilderMethod_string)(void*, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_string_length)(void*, const char*, int64_t);
  typedef ArrayBuilderError (*ArrayBuilderMethod_begin_list)(void*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_end_list)(void*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_begin_tuple)(void*, int64_t);
  typedef ArrayBuilderError (*ArrayBuilderMethod_index)(void*, int64_t);
  typedef ArrayBuilderError (*ArrayBuilderMethod_end_tuple)(void*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_begin_record)(void*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_begin_record_fast)(void*, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_begin_record_check)(void*, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_field_fast)(void*, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_field_check)(void*, const char*);
  typedef ArrayBuilderError (*ArrayBuilderMethod_end_record)(void*);

  class ArrayBuilder {{
    public:
      ArrayBuilder(void* ptr) : ptr_(ptr) {{ }}

      ArrayBuilderError length(int64_t* out) const noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_length>(
             {ctypes.cast(libawkward.ArrayBuilder_length, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), out);
      }}

      ArrayBuilderError clear() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_clear>(
             {ctypes.cast(libawkward.ArrayBuilder_clear, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError boolean(bool x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_boolean>(
             {ctypes.cast(libawkward.ArrayBuilder_boolean, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError append(bool x) noexcept {{
        return boolean(x);
      }}

      ArrayBuilderError integer(int64_t x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_integer>(
             {ctypes.cast(libawkward.ArrayBuilder_integer, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError append(int64_t x) noexcept {{
        return integer(x);
      }}

      ArrayBuilderError real(double x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_real>(
             {ctypes.cast(libawkward.ArrayBuilder_real, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError append(double x) noexcept {{
        return real(x);
      }}

      ArrayBuilderError complex(std::complex<double> x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_complex>(
             {ctypes.cast(libawkward.ArrayBuilder_complex, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x.real(), x.imag());
      }}

      ArrayBuilderError append(std::complex<double> x) noexcept {{
        return complex(x);
      }}

      // TODO: recognize std::chrono::time_point (what about units?)

      ArrayBuilderError datetime(int64_t x, const char* unit) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_datetime>(
             {ctypes.cast(libawkward.ArrayBuilder_datetime, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, unit);
      }}

      // TODO: recognize std::chrono::duration (what about units?)

      ArrayBuilderError timedelta(int64_t x, const char* unit) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_timedelta>(
             {ctypes.cast(libawkward.ArrayBuilder_timedelta, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, unit);
      }}

      ArrayBuilderError bytestring(const char* x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_bytestring>(
             {ctypes.cast(libawkward.ArrayBuilder_bytestring, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError bytestring_length(const char* x, int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_bytestring_length>(
             {ctypes.cast(libawkward.ArrayBuilder_bytestring_length, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, length);
      }}

      ArrayBuilderError bytestring(std::string x) noexcept {{
        return bytestring_length(x.c_str(), x.length());
      }}

      ArrayBuilderError string(const char* x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_string>(
             {ctypes.cast(libawkward.ArrayBuilder_string, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError string_length(const char* x, int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_string_length>(
             {ctypes.cast(libawkward.ArrayBuilder_string_length, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, length);
      }}

      ArrayBuilderError string(std::string x) noexcept {{
        return string_length(x.c_str(), x.length());
      }}

      ArrayBuilderError append(const char* x) noexcept {{
        return string(x);
      }}

      ArrayBuilderError append(const char* x, int64_t length) noexcept {{
        return string_length(x, length);
      }}

      ArrayBuilderError append(std::string x) noexcept {{
        return string(x);
      }}

      ArrayBuilderError begin_list() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_list>(
             {ctypes.cast(libawkward.ArrayBuilder_beginlist, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError end_list() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_end_list>(
             {ctypes.cast(libawkward.ArrayBuilder_endlist, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError begin_tuple(int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_tuple>(
             {ctypes.cast(libawkward.ArrayBuilder_begintuple, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), length);
      }}

      ArrayBuilderError index(int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_index>(
             {ctypes.cast(libawkward.ArrayBuilder_index, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), length);
      }}

      ArrayBuilderError end_tuple() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_end_tuple>(
             {ctypes.cast(libawkward.ArrayBuilder_endtuple, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError begin_record() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_record>(
             {ctypes.cast(libawkward.ArrayBuilder_beginrecord, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError begin_record_fast(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_record_fast>(
             {ctypes.cast(libawkward.ArrayBuilder_beginrecord_fast, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError begin_record_check(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_record_check>(
             {ctypes.cast(libawkward.ArrayBuilder_beginrecord_check, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError field_fast(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_field_fast>(
             {ctypes.cast(libawkward.ArrayBuilder_field_fast, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError field_check(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_field_check>(
             {ctypes.cast(libawkward.ArrayBuilder_field_check, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError end_record() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_end_record>(
             {ctypes.cast(libawkward.ArrayBuilder_endrecord, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

    private:
      void* ptr_;
  }};
}}
""".strip()
        cache[key] = out
        compiler(out)

    return out


def togenerator(form, flatlist_as_rvec):
    if isinstance(form, ak.forms.EmptyForm):
        return togenerator(form.to_NumpyForm(np.dtype(np.float64)), flatlist_as_rvec)

    elif isinstance(form, ak.forms.NumpyForm):
        if len(form.inner_shape) == 0:
            return NumpyArrayGenerator.from_form(form, flatlist_as_rvec)
        else:
            return togenerator(form.to_RegularForm(), flatlist_as_rvec)

    elif isinstance(form, ak.forms.RegularForm):
        return RegularArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, (ak.forms.ListForm, ak.forms.ListOffsetForm)):
        return ListArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.IndexedForm):
        return IndexedArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.IndexedOptionForm):
        return IndexedOptionArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.ByteMaskedForm):
        return ByteMaskedArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.BitMaskedForm):
        return BitMaskedArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.UnmaskedForm):
        return UnmaskedArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.RecordForm):
        return RecordArrayGenerator.from_form(form, flatlist_as_rvec)

    elif isinstance(form, ak.forms.UnionForm):
        return UnionArrayGenerator.from_form(form, flatlist_as_rvec)

    else:
        raise ak._errors.wrap_error(AssertionError(f"unrecognized Form: {type(form)}"))


class Generator:
    def IndexOf(self, arraytype):
        if arraytype == "int8_t":
            return ak.index.Index8
        elif arraytype == "uint8_t":
            return ak.index.IndexU8
        elif arraytype == "int32_t":
            return ak.index.Index32
        elif arraytype == "uint32_t":
            return ak.index.IndexU32
        elif arraytype == "int64_t":
            return ak.index.Index64
        elif arraytype == "uint64_t":
            return ak.index.IndexU64
        else:
            raise ak._errors.wrap_error(AssertionError(arraytype))

    def class_type_suffix(self, key):
        return ak._util.identifier_hash(key)

    def _generate_common(self, key):
        params = [
            f"if (parameter == {json.dumps(name)}) return {json.dumps(json.dumps(value))};\n      "
            for name, value in self.parameters.items()
        ]

        return f"""
        const std::string parameter(const std::string& parameter) const noexcept {{
      {"" if len(params) == 0 else "".join(x for x in params)}return "null";
    }}

    bool operator==({self.class_type()} other) const noexcept {{
      return start_ == other.start_  &&
             stop_ == other.stop_  &&
             which_ == other.which_  &&
             ptrs_ == other.ptrs_;
    }}

    bool operator!=({self.class_type()} other) const noexcept {{
      return start_ != other.start_  ||
             stop_ != other.stop_  ||
             which_ != other.which_  ||
             ptrs_ != other.ptrs_;
    }}

    Iterator<{self.class_type()}, value_type> begin() const noexcept {{
      return Iterator<{self.class_type()}, value_type>(start_, stop_, which_, ptrs_, lookup_);
    }}

    Iterator<{self.class_type()}, value_type> end() const noexcept {{
      return Iterator<{self.class_type()}, value_type>(stop_, stop_, which_, ptrs_, lookup_);
    }}

    RIterator<{self.class_type()}, value_type> rbegin() const noexcept {{
      return RIterator<{self.class_type()}, value_type>(stop_ - 1, stop_, which_, ptrs_, lookup_);
    }}

    RIterator<{self.class_type()}, value_type> rend() const noexcept {{
      return RIterator<{self.class_type()}, value_type>(start_ - 1, stop_, which_, ptrs_, lookup_);
    }}
        """.strip()

    def dataset(self, length="length", ptrs="ptrs"):
        return f"awkward::{self.class_type()}(0, {length}, 0, reinterpret_cast<ssize_t*>({ptrs}), 0)"

    def entry_type(self):
        return self.value_type()

    def entry(self, length="length", ptrs="ptrs", entry="i"):
        return f"{self.dataset(length=length, ptrs=ptrs)}[{entry}]"


class NumpyArrayGenerator(Generator, ak._lookup.NumpyLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return NumpyArrayGenerator(
            form.primitive,
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, primitive, parameters, flatlist_as_rvec):
        self.primitive = primitive
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.primitive,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.primitive == other.primitive
            and self.parameters == other.parameters
        )

    def class_type(self):
        return f"NumpyArray_{self.primitive}_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return {
            "bool": "bool",
            "int8": "int8_t",
            "uint8": "uint8_t",
            "int16": "int16_t",
            "uint16": "uint16_t",
            "int32": "int32_t",
            "uint32": "uint32_t",
            "int64": "int64_t",
            "uint64": "uint64_t",
            "float32": "float",
            "float64": "double",
            "complex64": "std::complex<float>",
            "complex128": "std::complex<double>",
            "datetime64": "std::time_point",
            "timedelta64": "std::duration",
        }[self.primitive]

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      return reinterpret_cast<{self.value_type()}*>(ptrs_[which_ + {self.ARRAY}])[start_ + at];
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RegularArrayGenerator(Generator, ak._lookup.RegularLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return RegularArrayGenerator(
            togenerator(form.content, flatlist_as_rvec),
            form.size,
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, content, size, parameters, flatlist_as_rvec):
        self.contenttype = content
        self.size = size
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        assert self.flatlist_as_rvec == self.contenttype.flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.contenttype,
                self.size,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contenttype == other.contenttype
            and self.size == other.size
            and self.parameters == other.parameters
        )

    @property
    def is_string(self):
        return isinstance(self.parameters, dict) and self.parameters.get(
            "__array__"
        ) in ("string", "bytestring")

    def is_flatlist(self):
        return isinstance(self.contenttype, NumpyArrayGenerator)

    def class_type(self):
        return f"RegularArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        if self.flatlist_as_rvec and self.is_flatlist:
            nested_type = self.contenttype.value_type()
            return f"ROOT::VecOps::RVec<{nested_type}>"
        else:
            return self.contenttype.class_type()

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        if not self.is_string and not (self.flatlist_as_rvec and self.is_flatlist):
            self.contenttype.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            if self.is_string:
                out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef std::string value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = (start_ + at) * {self.size};
      ssize_t stop = start + {self.size};
      ssize_t which = ptrs_[which_ + {self.CONTENT}];
      char* content = reinterpret_cast<char*>(ptrs_[which + {NumpyArrayGenerator.ARRAY}]) + start;
      return value_type(content, stop - start);
    }}
  }};
}}
""".strip()
            elif self.flatlist_as_rvec and self.is_flatlist:
                nested_type = self.contenttype.value_type()
                value_type = f"ROOT::VecOps::RVec<{nested_type}>"
                out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {value_type} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = (start_ + at) * {self.size};
      ssize_t stop = start + {self.size};
      ssize_t which = ptrs_[which_ + {self.CONTENT}];
      {nested_type}* content = reinterpret_cast<{nested_type}*>(ptrs_[which + {NumpyArrayGenerator.ARRAY}]) + start;
      return value_type(content, stop - start);
    }}
  }};
}}
""".strip()
            else:
                out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = (start_ + at) * {self.size};
      ssize_t stop = start + {self.size};
      return value_type(start, stop, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class ListArrayGenerator(Generator, ak._lookup.ListLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        if isinstance(form, ak.forms.ListForm):
            index_string = form.starts
        else:
            index_string = form.offsets

        return ListArrayGenerator(
            index_string,
            togenerator(form.content, flatlist_as_rvec),
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, index_type, content, parameters, flatlist_as_rvec):
        if index_type == "i32":
            self.index_type = "int32_t"
        elif index_type == "u32":
            self.index_type = "uint32_t"
        elif index_type == "i64":
            self.index_type = "int64_t"
        else:
            raise ak._errors.wrap_error(AssertionError(index_type))
        self.content = content

        # FIXME: satisfy the ContentLookup super-class
        self.contenttype = content
        self.indextype = self.index_type

        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        assert self.flatlist_as_rvec == self.content.flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.index_type,
                self.content,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.index_type == other.index_type
            and self.content == other.content
            and self.parameters == other.parameters
        )

    @property
    def is_string(self):
        return isinstance(self.parameters, dict) and self.parameters.get(
            "__array__"
        ) in ("string", "bytestring")

    @property
    def is_flatlist(self):
        return isinstance(self.content, NumpyArrayGenerator)

    def class_type(self):
        return f"ListArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        if self.is_string:
            return "std::string"
        elif self.flatlist_as_rvec and self.is_flatlist:
            nested_type = self.content.value_type()
            return f"ROOT::VecOps::RVec<{nested_type}>"
        else:
            return self.content.class_type()

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        if not self.is_string and not (self.flatlist_as_rvec and self.is_flatlist):
            self.content.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            if self.is_string:
                out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef std::string value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STARTS}])[start_ + at];
      ssize_t stop = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STOPS}])[start_ + at];
      ssize_t which = ptrs_[which_ + {self.CONTENT}];
      char* content = reinterpret_cast<char*>(ptrs_[which + {NumpyArrayGenerator.ARRAY}]) + start;
      return value_type(content, stop - start);
    }}
  }};
}}
""".strip()
            elif self.flatlist_as_rvec and self.is_flatlist:
                nested_type = self.content.value_type()
                value_type = f"ROOT::VecOps::RVec<{nested_type}>"
                out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {value_type} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STARTS}])[start_ + at];
      ssize_t stop = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STOPS}])[start_ + at];
      ssize_t which = ptrs_[which_ + {self.CONTENT}];
      {nested_type}* content = reinterpret_cast<{nested_type}*>(ptrs_[which + {NumpyArrayGenerator.ARRAY}]) + start;
      return value_type(content, stop - start);
    }}
  }};
}}
""".strip()
            else:
                out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STARTS}])[start_ + at];
      ssize_t stop = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STOPS}])[start_ + at];
      return value_type(start, stop, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class IndexedArrayGenerator(Generator, ak._lookup.IndexedLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return IndexedArrayGenerator(
            form.index,
            togenerator(form.content, flatlist_as_rvec),
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, index_type, content, parameters, flatlist_as_rvec):
        if index_type == "i32":
            self.indextype = "int32_t"
        elif index_type == "u32":
            self.indextype = "uint32_t"
        elif index_type == "i64":
            self.indextype = "int64_t"
        else:
            raise ak._errors.wrap_error(AssertionError(index_type))
        self.contenttype = content
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        assert self.flatlist_as_rvec == self.contenttype.flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.indextype,
                self.contenttype,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.indextype == other.indextype
            and self.contenttype == other.contenttype
            and self.parameters == other.parameters
        )

    def class_type(self):
        return f"IndexedArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return self.contenttype.value_type()

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.contenttype.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t index = reinterpret_cast<{self.indextype}*>(ptrs_[which_ + {self.INDEX}])[start_ + at];
      return {self.contenttype.class_type()}(index, index + 1, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_)[0];
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class IndexedOptionArrayGenerator(Generator, ak._lookup.IndexedOptionLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return IndexedOptionArrayGenerator(
            form.index,
            togenerator(form.content, flatlist_as_rvec),
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, index_type, content, parameters, flatlist_as_rvec):
        if index_type == "i32":
            self.index_type = "int32_t"
        elif index_type == "i64":
            self.index_type = "int64_t"
        else:
            raise ak._errors.wrap_error(AssertionError(index_type))
        self.contenttype = content
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        self.indextype = self.index_type
        assert self.flatlist_as_rvec == self.contenttype.flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.index_type,
                self.contenttype,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.index_type == other.index_type
            and self.contenttype == other.contenttype
            and self.parameters == other.parameters
        )

    def class_type(self):
        return f"IndexedOptionArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return f"std::optional<{self.contenttype.value_type()}>"

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.contenttype.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t index = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.INDEX}])[start_ + at];
      if (index >= 0) {{
        return value_type{{ {self.contenttype.class_type()}(index, index + 1, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_)[0] }};
      }}
      else {{
        return std::nullopt;
      }}
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class ByteMaskedArrayGenerator(Generator, ak._lookup.ByteMaskedLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return ByteMaskedArrayGenerator(
            togenerator(form.content, flatlist_as_rvec),
            form.valid_when,
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, content, valid_when, parameters, flatlist_as_rvec):
        self.contenttype = content
        self.valid_when = valid_when
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        assert self.flatlist_as_rvec == self.contenttype.flatlist_as_rvec
        self.masktype = "int8_t"

    def __hash__(self):
        return hash(
            (
                type(self),
                self.contenttype,
                self.valid_when,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contenttype == other.contenttype
            and self.valid_when == other.valid_when
            and self.parameters == other.parameters
        )

    def class_type(self):
        return (
            f"ByteMaskedArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"
        )

    def value_type(self):
        return f"std::optional<{self.contenttype.value_type()}>"

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.contenttype.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      int8_t mask = reinterpret_cast<int8_t*>(ptrs_[which_ + {self.MASK}])[start_ + at];
      if ({"mask != 0" if self.valid_when else "mask == 0"}) {{
        return value_type{{ {self.contenttype.class_type()}(start_, stop_, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_)[at] }};
      }}
      else {{
        return std::nullopt;
      }}
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class BitMaskedArrayGenerator(Generator, ak._lookup.BitMaskedLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return BitMaskedArrayGenerator(
            togenerator(form.content, flatlist_as_rvec),
            form.valid_when,
            form.lsb_order,
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(
        self,
        content,
        valid_when,
        lsb_order,
        parameters,
        flatlist_as_rvec,
    ):
        self.contenttype = content
        self.valid_when = valid_when
        self.lsb_order = lsb_order
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        assert self.flatlist_as_rvec == self.contenttype.flatlist_as_rvec
        self.masktype = "uint8_t"

    def __hash__(self):
        return hash(
            (
                self.contenttype,
                self.valid_when,
                self.lsb_order,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contenttype == other.contenttype
            and self.valid_when == other.valid_when
            and self.lsb_order == other.lsb_order
            and self.parameters == other.parameters
        )

    def class_type(self):
        return f"BitMaskedArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return f"std::optional<{self.contenttype.value_type()}>"

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.contenttype.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      size_t startat = start_ + at;
      size_t bitat = startat / 8;
      size_t shift = startat % 8;
      uint8_t byte = reinterpret_cast<uint8_t*>(ptrs_[which_ + {self.MASK}])[bitat];
      uint8_t mask = {"(byte >> shift) & 1" if self.lsb_order else "(byte << shift) & 128"};

      if ({"mask != 0" if self.valid_when else "mask == 0"}) {{
        return value_type{{ {self.contenttype.class_type()}(start_, stop_, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_)[at] }};
      }}
      else {{
        return std::nullopt;
      }}
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class UnmaskedArrayGenerator(Generator, ak._lookup.UnmaskedLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return UnmaskedArrayGenerator(
            togenerator(form.content, flatlist_as_rvec),
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, content, parameters, flatlist_as_rvec):
        self.contenttype = content
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        assert self.flatlist_as_rvec == self.contenttype.flatlist_as_rvec

    def __hash__(self):
        return hash((type(self), self.contenttype, json.dumps(self.parameters)))

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contenttype == other.contenttype
            and self.parameters == other.parameters
        )

    def class_type(self):
        return f"UnmaskedArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return f"std::optional<{self.contenttype.value_type()}>"

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.contenttype.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      return value_type{{ {self.contenttype.class_type()}(start_, stop_, ptrs_[which_ + {self.CONTENT}], ptrs_, lookup_)[at] }};
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RecordGenerator(Generator, ak._lookup.RecordLookup):
    def __init__(self, contents, fields, parameters, flatlist_as_rvec):
        self.contents = tuple(contents)
        self.fields = None if fields is None else tuple(fields)
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        for content in self.contents:
            assert self.flatlist_as_rvec == content.flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.contents,
                self.fields,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contents == other.contents
            and self.fields == other.fields
            and self.parameters == other.parameters
        )

    def class_type(self):
        if (
            isinstance(self.parameters, dict)
            and self.parameters.get("__record__") is not None
        ):
            insert = "_" + self.parameters["__record__"]
        else:
            insert = ""
        return f"Record{insert}_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def generate(self, compiler, use_cached=True):
        generate_RecordView(compiler, use_cached=use_cached)
        for content in self.contents:
            content.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            params = [
                f"if (parameter == {json.dumps(name)}) return {json.dumps(json.dumps(value))};\n      "
                for name, value in self.parameters.items()
            ]

            if self.fields is None:
                fieldnames = [f"slot{i}" for i in range(len(self.contents))]
            else:
                fieldnames = self.fields
            getfields = []
            for i, (fieldname, content) in enumerate(zip(fieldnames, self.contents)):
                if re.match("^[A-Za-z_][A-Za-z_0-9]*$", fieldname) is not None:
                    getfields.append(
                        f"""
    {content.value_type()} {fieldname}() const noexcept {{
      return {content.class_type()}(at_, at_ + 1, ptrs_[which_ + {self.CONTENTS + i}], ptrs_, lookup_)[0];
    }}
""".strip()
                    )

            eoln = "\n    "
            out = f"""
namespace awkward {{
  class {self.class_type()}: public RecordView {{
  public:
    {self.class_type()}(ssize_t at, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : RecordView(at, which, ptrs, lookup) {{ }}
    {self.class_type()}() : RecordView(0, 0, 0, 0) {{ }}

    const std::string parameter(const std::string& parameter) const noexcept {{
      {"" if len(params) == 0 else "".join(x for x in params)}return "null";
    }}

    {eoln.join(getfields)}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RecordArrayGenerator(Generator, ak._lookup.RecordLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return RecordArrayGenerator(
            [togenerator(x, flatlist_as_rvec) for x in form.contents],
            None if form.is_tuple else form.fields,
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, contents, fields, parameters, flatlist_as_rvec):
        self.contenttypes = tuple(contents)
        self.fields = None if fields is None else tuple(fields)
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        for content in self.contenttypes:
            assert self.flatlist_as_rvec == content.flatlist_as_rvec

        self.record = RecordGenerator(contents, fields, parameters, flatlist_as_rvec)

    def __hash__(self):
        return hash(
            (
                type(self),
                self.contenttypes,
                self.fields,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contenttypes == other.contenttypes
            and self.fields == other.fields
            and self.parameters == other.parameters
        )

    def class_type(self):
        if (
            isinstance(self.parameters, dict)
            and self.parameters.get("__record__") is not None
        ):
            insert = "_" + self.parameters["__record__"]
        else:
            insert = ""
        return f"RecordArray{insert}_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return self.record.class_type()

    def fieldindex(self, key):
        out = -1
        if self.fields is not None:
            for i, x in enumerate(self.fields):
                if x == key:
                    out = i
                    break
        if out == -1:
            try:
                out = int(key)
            except ValueError:
                return None
            if not 0 <= out < len(self.contenttypes):
                return None
        return out

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.record.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      return value_type(start_ + at, which_, ptrs_, lookup_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class UnionArrayGenerator(Generator, ak._lookup.UnionLookup):
    @classmethod
    def from_form(cls, form, flatlist_as_rvec):
        return UnionArrayGenerator(
            form.index,
            [togenerator(x, flatlist_as_rvec) for x in form.contents],
            form.parameters,
            flatlist_as_rvec,
        )

    def __init__(self, index_type, contents, parameters, flatlist_as_rvec):
        self.tagstype = "int8_t"
        if index_type == "i32":
            self.indextype = "int32_t"
        elif index_type == "u32":
            self.indextype = "uint32_t"
        elif index_type == "i64":
            self.indextype = "int64_t"
        else:
            raise ak._errors.wrap_error(AssertionError(index_type))
        self.contenttypes = tuple(contents)
        self.parameters = parameters
        self.flatlist_as_rvec = flatlist_as_rvec
        for content in self.contenttypes:
            assert self.flatlist_as_rvec == content.flatlist_as_rvec

    def __hash__(self):
        return hash(
            (
                type(self),
                self.indextype,
                self.contenttypes,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.indextype == other.indextype
            and self.contenttypes == other.contenttypes
            and self.parameters == other.parameters
        )

    def class_type(self):
        return f"UnionArray_{self.class_type_suffix((self, self.flatlist_as_rvec))}"

    def value_type(self):
        return f"std::variant<{','.join(x.value_type() for x in self.contenttypes)}>"

    def generate(self, compiler, use_cached=True):
        generate_ArrayView(compiler, use_cached=use_cached)
        for content in self.contenttypes:
            content.generate(compiler, use_cached)

        key = (self, self.flatlist_as_rvec)
        if not use_cached or key not in cache:
            cases = []
            for i, content in enumerate(self.contenttypes):
                cases.append(
                    f"""
        case {i}:
          return value_type{{ {content.class_type()}(index, index + 1, ptrs_[which_ + {self.CONTENTS + i}], ptrs_, lookup_)[0] }};
""".strip()
                )

            eoln = "\n        "
            out = f"""
namespace awkward {{
  class {self.class_type()}: public ArrayView {{
  public:
    {self.class_type()}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs, PyObject* lookup)
      : ArrayView(start, stop, which, ptrs, lookup) {{ }}
    {self.class_type()}() : ArrayView(0, 0, 0, 0, 0) {{ }}

    typedef {self.value_type()} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      int8_t tag = reinterpret_cast<int8_t*>(ptrs_[which_ + {self.TAGS}])[start_ + at];
      {self.indextype} index = reinterpret_cast<{self.indextype}*>(ptrs_[which_ + {self.INDEX}])[start_ + at];
      switch (tag) {{
        {eoln.join(cases)}
        default:
          return value_type{{ {self.contenttypes[0].class_type()}(index, index + 1, ptrs_[which_ + {self.CONTENTS + 0}], ptrs_, lookup_)[0] }};
      }}
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)

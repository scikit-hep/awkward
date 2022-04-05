# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import base64
import ctypes
import struct
import json
import re
import threading

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


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
    Iterator(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs) { }

    VALUE operator*() const noexcept {
      return ARRAY(start_, stop_, which_, ptrs_)[0];
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
  };

  template <typename ARRAY, typename VALUE>
  class RIterator {
  public:
    RIterator(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs) { }

    VALUE operator*() const noexcept {
      return ARRAY(start_, stop_, which_, ptrs_)[0];
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
  };

  class ArrayView {
  public:
    ArrayView(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs) {
      cout << "...ArrayView " << this << " is constructed." << endl;
    }

    ~ArrayView() {
        cout << "...ArrayView " << this << " is destructed!" << endl;
    }

    size_t size() const noexcept {
      return stop_ - start_;
    }

    bool empty() const noexcept {
      return start_ == stop_;
    }

  protected:
    ssize_t start_;
    ssize_t stop_;
    ssize_t which_;
    ssize_t* ptrs_;
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
    RecordView(ssize_t at, ssize_t which, ssize_t* ptrs)
      : at_(at), which_(which), ptrs_(ptrs) { }

  protected:
    ssize_t at_;
    ssize_t which_;
    ssize_t* ptrs_;
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
             {ctypes.cast(ak._libawkward.ArrayBuilder_length, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), out);
      }}

      ArrayBuilderError clear() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_clear>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_clear, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError boolean(bool x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_boolean>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_boolean, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError append(bool x) noexcept {{
        return boolean(x);
      }}

      ArrayBuilderError integer(int64_t x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_integer>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_integer, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError append(int64_t x) noexcept {{
        return integer(x);
      }}

      ArrayBuilderError real(double x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_real>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_real, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError append(double x) noexcept {{
        return real(x);
      }}

      ArrayBuilderError complex(std::complex<double> x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_complex>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_complex, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x.real(), x.imag());
      }}

      ArrayBuilderError append(std::complex<double> x) noexcept {{
        return complex(x);
      }}

      // TODO: recognize std::chrono::time_point (what about units?)

      ArrayBuilderError datetime(int64_t x, const char* unit) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_datetime>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_datetime, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, unit);
      }}

      // TODO: recognize std::chrono::duration (what about units?)

      ArrayBuilderError timedelta(int64_t x, const char* unit) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_timedelta>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_timedelta, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, unit);
      }}

      ArrayBuilderError bytestring(const char* x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_bytestring>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_bytestring, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError bytestring_length(const char* x, int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_bytestring_length>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_bytestring_length, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x, length);
      }}

      ArrayBuilderError bytestring(std::string x) noexcept {{
        return bytestring_length(x.c_str(), x.length());
      }}

      ArrayBuilderError string(const char* x) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_string>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_string, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), x);
      }}

      ArrayBuilderError string_length(const char* x, int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_string_length>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_string_length, ctypes.c_voidp).value}
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
             {ctypes.cast(ak._libawkward.ArrayBuilder_beginlist, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError end_list() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_end_list>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_endlist, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError begin_tuple(int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_tuple>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_begintuple, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), length);
      }}

      ArrayBuilderError index(int64_t length) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_index>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_index, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), length);
      }}

      ArrayBuilderError end_tuple() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_end_tuple>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_endtuple, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError begin_record() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_record>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_));
      }}

      ArrayBuilderError begin_record_fast(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_record_fast>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord_fast, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError begin_record_check(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_begin_record_check>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord_check, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError field_fast(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_field_fast>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_field_fast, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError field_check(const char* name) noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_field_check>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_field_check, ctypes.c_voidp).value}
        )(reinterpret_cast<void*>(ptr_), name);
      }}

      ArrayBuilderError end_record() noexcept {{
        return reinterpret_cast<ArrayBuilderMethod_end_record>(
             {ctypes.cast(ak._libawkward.ArrayBuilder_endrecord, ctypes.c_voidp).value}
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


def togenerator(form):
    if isinstance(form, ak._v2.forms.EmptyForm):
        return togenerator(form.toNumpyForm(np.dtype(np.float64)))

    elif isinstance(form, ak._v2.forms.NumpyForm):
        if len(form.inner_shape) == 0:
            return NumpyArrayGenerator.from_form(form)
        else:
            return togenerator(form.toRegularForm())

    elif isinstance(form, ak._v2.forms.RegularForm):
        return RegularArrayGenerator.from_form(form)

    elif isinstance(form, (ak._v2.forms.ListForm, ak._v2.forms.ListOffsetForm)):
        return ListArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.IndexedForm):
        return IndexedArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.IndexedOptionForm):
        return IndexedOptionArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.ByteMaskedForm):
        return ByteMaskedArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.BitMaskedForm):
        return BitMaskedArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.UnmaskedForm):
        return UnmaskedArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.RecordForm):
        return RecordArrayGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.UnionForm):
        return UnionArrayGenerator.from_form(form)

    else:
        raise ak._v2._util.error(AssertionError(f"unrecognized Form: {type(form)}"))


class Generator:
    @classmethod
    def form_from_identifier(cls, form):
        if not form.has_identifier:
            return None
        else:
            raise ak._v2._util.error(NotImplementedError("TODO: identifiers in C++"))

    def class_type_suffix(self, key):
        return (
            base64.encodebytes(struct.pack("q", hash(key)))
            .rstrip(b"=\n")
            .replace(b"+", b"")
            .replace(b"/", b"")
            .decode("ascii")
        )

    def _generate_common(self, key):
        params = [
            f"if (parameter == {json.dumps(name)}) return {json.dumps(json.dumps(value))};\n      "
            for name, value in self.parameters.items()
        ]

        return f"""
        const std::string parameter(const std::string& parameter) const noexcept {{
      {"" if len(params) == 0 else "".join(x for x in params)}return "null";
    }}

    bool operator==({self.class_type(key[1:])} other) const noexcept {{
      return start_ == other.start_  &&
             stop_ == other.stop_  &&
             which_ == other.which_  &&
             ptrs_ == other.ptrs_;
    }}

    bool operator!=({self.class_type(key[1:])} other) const noexcept {{
      return start_ != other.start_  ||
             stop_ != other.stop_  ||
             which_ != other.which_  ||
             ptrs_ != other.ptrs_;
    }}

    Iterator<{self.class_type(key[1:])}, value_type> begin() const noexcept {{
      return Iterator<{self.class_type(key[1:])}, value_type>(start_, stop_, which_, ptrs_);
    }}

    Iterator<{self.class_type(key[1:])}, value_type> end() const noexcept {{
      return Iterator<{self.class_type(key[1:])}, value_type>(stop_, stop_, which_, ptrs_);
    }}

    RIterator<{self.class_type(key[1:])}, value_type> rbegin() const noexcept {{
      return RIterator<{self.class_type(key[1:])}, value_type>(stop_ - 1, stop_, which_, ptrs_);
    }}

    RIterator<{self.class_type(key[1:])}, value_type> rend() const noexcept {{
      return RIterator<{self.class_type(key[1:])}, value_type>(start_ - 1, stop_, which_, ptrs_);
    }}
        """.strip()

    def dataset(self, length="length", ptrs="ptrs", flatlist_as_rvec=False):
        key = (self, flatlist_as_rvec)
        return f"awkward::{self.class_type(key[1:])}(0, {length}, 0, reinterpret_cast<ssize_t*>({ptrs}))"

    def entry_type(self, flatlist_as_rvec=False):
        key = (self, flatlist_as_rvec)
        return self.value_type(key[1:])

    def entry(self, length="length", ptrs="ptrs", entry="i", flatlist_as_rvec=False):
        return f"{self.dataset(length=length, ptrs=ptrs, flatlist_as_rvec=flatlist_as_rvec)}[{entry}]"


class NumpyArrayGenerator(Generator, ak._v2._lookup.NumpyLookup):
    @classmethod
    def from_form(cls, form):
        return NumpyArrayGenerator(
            form.primitive, cls.form_from_identifier(form), form.parameters
        )

    def __init__(self, primitive, identifier, parameters):
        self.primitive = primitive
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.primitive,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.primitive == other.primitive
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"NumpyArray_{self.primitive}_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
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

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      return reinterpret_cast<{self.value_type(key[1:])}*>(ptrs_[which_ + {self.ARRAY}])[start_ + at];
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RegularArrayGenerator(Generator, ak._v2._lookup.RegularLookup):
    @classmethod
    def from_form(cls, form):
        return RegularArrayGenerator(
            togenerator(form.content),
            form.size,
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, content, size, identifier, parameters):
        self.content = content
        self.size = size
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.content,
                self.size,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.content == other.content
            and self.size == other.size
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    @property
    def is_string(self):
        return isinstance(self.parameters, dict) and self.parameters.get(
            "__array__"
        ) in ("string", "bytestring")

    def is_flatlist(self):
        return isinstance(self.content, NumpyArrayGenerator)

    def class_type(self, key):
        return f"RegularArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return self.content.class_type(key)

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        if not self.is_string and not (flatlist_as_rvec and self.is_flatlist):
            self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            if self.is_string:
                out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

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
            elif flatlist_as_rvec and self.is_flatlist:
                nested_type = self.content.value_type(key[1:])
                value_type = f"ROOT::RVec<{nested_type}>"
                out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

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
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = (start_ + at) * {self.size};
      ssize_t stop = start + {self.size};
      return value_type(start, stop, ptrs_[which_ + {self.CONTENT}], ptrs_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class ListArrayGenerator(Generator, ak._v2._lookup.ListLookup):
    @classmethod
    def from_form(cls, form):
        if isinstance(form, ak._v2.forms.ListForm):
            index_string = form.starts
        else:
            index_string = form.offsets

        return ListArrayGenerator(
            index_string,
            togenerator(form.content),
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, index_type, content, identifier, parameters):
        if index_type == "i32":
            self.index_type = "int32_t"
        elif index_type == "u32":
            self.index_type = "uint32_t"
        elif index_type == "i64":
            self.index_type = "int64_t"
        else:
            raise ak._v2._util.error(AssertionError(index_type))
        self.content = content
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.index_type,
                self.content,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.index_type == other.index_type
            and self.content == other.content
            and self.identifier == other.identifier
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

    def class_type(self, key):
        return f"ListArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return self.content.class_type(key)

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        if not self.is_string and not (flatlist_as_rvec and self.is_flatlist):
            self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            if self.is_string:
                out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

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
            elif flatlist_as_rvec and self.is_flatlist:
                nested_type = self.content.value_type(key[1:])
                value_type = f"ROOT::RVec<{nested_type}>"
                out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

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
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{
      cout << "......{self.class_type(key[1:])} " << this << " is constructed!" << endl;
    }}

    ~{self.class_type(key[1:])}() {{
        cout << "......{self.class_type(key[1:])} " << this << " is destructed!" << endl;
    }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STARTS}])[start_ + at];
      ssize_t stop = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STOPS}])[start_ + at];
      return value_type(start, stop, ptrs_[which_ + {self.CONTENT}], ptrs_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class IndexedArrayGenerator(Generator, ak._v2._lookup.IndexedLookup):
    @classmethod
    def from_form(cls, form):
        return IndexedArrayGenerator(
            form.index,
            togenerator(form.content),
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, index_type, content, identifier, parameters):
        if index_type == "i32":
            self.index_type = "int32_t"
        elif index_type == "u32":
            self.index_type = "uint32_t"
        elif index_type == "i64":
            self.index_type = "int64_t"
        else:
            raise ak._v2._util.error(AssertionError(index_type))
        self.content = content
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.index_type,
                self.content,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.index_type == other.index_type
            and self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"IndexedArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return self.content.value_type(key)

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t index = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.INDEX}])[start_ + at];
      return {self.content.class_type(key[1:])}(index, index + 1, ptrs_[which_ + {self.CONTENT}], ptrs_)[0];
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class IndexedOptionArrayGenerator(Generator, ak._v2._lookup.IndexedOptionLookup):
    @classmethod
    def from_form(cls, form):
        return IndexedOptionArrayGenerator(
            form.index,
            togenerator(form.content),
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, index_type, content, identifier, parameters):
        if index_type == "i32":
            self.index_type = "int32_t"
        elif index_type == "i64":
            self.index_type = "int64_t"
        else:
            raise ak._v2._util.error(AssertionError(index_type))
        self.content = content
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.index_type,
                self.content,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.index_type == other.index_type
            and self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"IndexedOptionArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return f"std::optional<{self.content.value_type(key)}>"

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      ssize_t index = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.INDEX}])[start_ + at];
      if (index >= 0) {{
        return value_type{{ {self.content.class_type(key[1:])}(index, index + 1, ptrs_[which_ + {self.CONTENT}], ptrs_)[0] }};
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


class ByteMaskedArrayGenerator(Generator, ak._v2._lookup.ByteMaskedLookup):
    @classmethod
    def from_form(cls, form):
        return ByteMaskedArrayGenerator(
            togenerator(form.content),
            form.valid_when,
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, content, valid_when, identifier, parameters):
        self.content = content
        self.valid_when = valid_when
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.content,
                self.valid_when,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.content == other.content
            and self.valid_when == other.valid_when
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"ByteMaskedArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return f"std::optional<{self.content.value_type(key)}>"

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      int8_t mask = reinterpret_cast<int8_t*>(ptrs_[which_ + {self.MASK}])[start_ + at];
      if ({"mask != 0" if self.valid_when else "mask == 0"}) {{
        return value_type{{ {self.content.class_type(key[1:])}(start_, stop_, ptrs_[which_ + {self.CONTENT}], ptrs_)[at] }};
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


class BitMaskedArrayGenerator(Generator, ak._v2._lookup.BitMaskedLookup):
    @classmethod
    def from_form(cls, form):
        return BitMaskedArrayGenerator(
            togenerator(form.content),
            form.valid_when,
            form.lsb_order,
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, content, valid_when, lsb_order, identifier, parameters):
        self.content = content
        self.valid_when = valid_when
        self.lsb_order = lsb_order
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                self.content,
                self.valid_when,
                self.lsb_order,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.content == other.content
            and self.valid_when == other.valid_when
            and self.lsb_order == other.lsb_order
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"BitMaskedArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return f"std::optional<{self.content.value_type(key)}>"

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      size_t startat = start_ + at;
      size_t bitat = startat / 8;
      size_t shift = startat % 8;
      uint8_t byte = reinterpret_cast<uint8_t*>(ptrs_[which_ + {self.MASK}])[bitat];
      uint8_t mask = {"(byte >> shift) & 1" if self.lsb_order else "(byte << shift) & 128"};

      if ({"mask != 0" if self.valid_when else "mask == 0"}) {{
        return value_type{{ {self.content.class_type(key[1:])}(start_, stop_, ptrs_[which_ + {self.CONTENT}], ptrs_)[at] }};
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


class UnmaskedArrayGenerator(Generator, ak._v2._lookup.UnmaskedLookup):
    @classmethod
    def from_form(cls, form):
        return UnmaskedArrayGenerator(
            togenerator(form.content), cls.form_from_identifier(form), form.parameters
        )

    def __init__(self, content, identifier, parameters):
        self.content = content
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (type(self), self.content, self.identifier, json.dumps(self.parameters))
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"UnmaskedArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return f"std::optional<{self.content.value_type(key)}>"

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      return value_type{{ {self.content.class_type(key[1:])}(start_, stop_, ptrs_[which_ + {self.CONTENT}], ptrs_)[at] }};
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RecordGenerator(Generator, ak._v2._lookup.RecordLookup):
    def __init__(self, contents, fields, parameters):
        self.contents = tuple(contents)
        self.fields = None if fields is None else tuple(fields)
        self.parameters = parameters

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

    def class_type(self, key):
        if (
            isinstance(self.parameters, dict)
            and self.parameters.get("__record__") is not None
        ):
            insert = "_" + self.parameters["__record__"]
        else:
            insert = ""
        return f"Record{insert}_{self.class_type_suffix((self,) + key)}"

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_RecordView(compiler, use_cached=use_cached)
        for content in self.contents:
            content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
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
    {content.value_type(key[1:])} {fieldname}() const noexcept {{
      return {content.class_type(key[1:])}(at_, at_ + 1, ptrs_[which_ + {self.CONTENTS + i}], ptrs_)[0];
    }}
""".strip()
                    )

            eoln = "\n    "
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public RecordView {{
  public:
    {self.class_type(key[1:])}(ssize_t at, ssize_t which, ssize_t* ptrs)
      : RecordView(at, which, ptrs) {{ }}

    const std::string parameter(const std::string& parameter) const noexcept {{
      {"" if len(params) == 0 else "".join(x for x in params)}return "null";
    }}

    {eoln.join(getfields)}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RecordArrayGenerator(Generator, ak._v2._lookup.RecordLookup):
    @classmethod
    def from_form(cls, form):
        return RecordArrayGenerator(
            [togenerator(x) for x in form.contents],
            None if form.is_tuple else form.fields,
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, contents, fields, identifier, parameters):
        self.contents = tuple(contents)
        self.fields = None if fields is None else tuple(fields)
        self.identifier = identifier
        self.parameters = parameters

        self.record = RecordGenerator(contents, fields, parameters)

    def __hash__(self):
        return hash(
            (
                type(self),
                self.contents,
                self.fields,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.contents == other.contents
            and self.fields == other.fields
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        if (
            isinstance(self.parameters, dict)
            and self.parameters.get("__record__") is not None
        ):
            insert = "_" + self.parameters["__record__"]
        else:
            insert = ""
        return f"RecordArray{insert}_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return self.record.class_type(key)

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.record.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      return value_type(start_ + at, which_, ptrs_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class UnionArrayGenerator(Generator, ak._v2._lookup.UnionLookup):
    @classmethod
    def from_form(cls, form):
        return UnionArrayGenerator(
            form.index,
            [togenerator(x) for x in form.contents],
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, index_type, contents, identifier, parameters):
        if index_type == "i32":
            self.index_type = "int32_t"
        elif index_type == "u32":
            self.index_type = "uint32_t"
        elif index_type == "i64":
            self.index_type = "int64_t"
        else:
            raise ak._v2._util.error(AssertionError(index_type))
        self.contents = tuple(contents)
        self.identifier = identifier
        self.parameters = parameters

    def __hash__(self):
        return hash(
            (
                type(self),
                self.index_type,
                self.contents,
                self.identifier,
                json.dumps(self.parameters),
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.index_type == other.index_type
            and self.contents == other.contents
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    def class_type(self, key):
        return f"UnionArray_{self.class_type_suffix((self,) + key)}"

    def value_type(self, key):
        return f"std::variant<{','.join(x.value_type(key) for x in self.contents)}>"

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        for content in self.contents:
            content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, flatlist_as_rvec)
        if not use_cached or key not in cache:
            cases = []
            for i, content in enumerate(self.contents):
                cases.append(
                    f"""
        case {i}:
          return value_type{{ {content.class_type(key[1:])}(index, index + 1, ptrs_[which_ + {self.CONTENTS + i}], ptrs_)[0] }};
""".strip()
                )

            eoln = "\n        "
            out = f"""
namespace awkward {{
  class {self.class_type(key[1:])}: public ArrayView {{
  public:
    {self.class_type(key[1:])}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common(key)}

    value_type operator[](size_t at) const noexcept {{
      int8_t tag = reinterpret_cast<int8_t*>(ptrs_[which_ + {self.TAGS}])[start_ + at];
      {self.index_type} index = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.INDEX}])[start_ + at];
      switch (tag) {{
        {eoln.join(cases)}
        default:
          return value_type{{ {self.contents[0].class_type(key[1:])}(index, index + 1, ptrs_[which_ + {self.CONTENTS + 0}], ptrs_)[0] }};
      }}
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class CppStatements:
    dtype_to_cpp = {
        np.dtype(np.int8): "char",
        np.dtype(np.int16): "int16_t",
        np.dtype(np.int32): "int32_t",
        np.dtype(np.int64): "int64_t",
        np.dtype(np.uint8): "unsigned char",
        np.dtype(np.uint16): "uint16_t",
        np.dtype(np.uint32): "uint32_t",
        np.dtype(np.uint64): "uint64_t",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }

    def __init__(self, cppcode, **kwargs):
        compiler = RawCppCompiler.instance()

        self._cppcode = cppcode
        self._funcname = f"awkward_function_{compiler.next_number()}"
        self._funcargs = []
        self._assignments = []
        self._order = []
        self._argname_to_index = {}
        self._forms = []
        self._generators = []
        for argname, argval in kwargs.items():
            self._order.append(argname)
            if isinstance(argval, (ak._v2.Array, ak._v2.forms.Form)):
                if isinstance(argval, ak._v2.Array):
                    form = argval.layout.form
                else:
                    form = argval
                number = compiler.next_number()
                length = f"awkward_argument_{number}_length"
                ptrs = f"awkward_argument_{number}_ptrs"
                self._funcargs.append(f"ssize_t {length}, ssize_t {ptrs}")
                generator = togenerator(form)
                generator.generate(compiler.declare)
                self._assignments.append(
                    f"auto {argname} = {generator.dataset(length, ptrs)};"
                )
                self._argname_to_index[argname] = len(self._forms)
                self._forms.append(form)
                self._generators.append(generator)

            elif isinstance(argval, ak._v2.ArrayBuilder):
                generate_ArrayBuilder(compiler.declare)
                number = compiler.next_number()
                ptr = f"awkward_argument_{number}_ptr"
                self._funcargs.append(f"ssize_t {ptr}")
                self._assignments.append(
                    f"auto {argname} = awkward::ArrayBuilder(reinterpret_cast<void*>({ptr}));"
                )
                self._argname_to_index[argname] = "ArrayBuilder"

            elif isinstance(argval, (numpy.ndarray, np.dtype)):
                if isinstance(argval, numpy.ndarray):
                    dtype = argval.dtype
                else:
                    dtype = argval
                if dtype not in self.dtype_to_cpp:
                    raise ak._v2._util.error(
                        TypeError(f"arrays of {dtype} not allowed")
                    )

                number = compiler.next_number()
                ptr = f"awkward_argument_{number}_ptr"
                self._funcargs.append(f"ssize_t {ptr}")
                self._assignments.append(
                    f"auto {argname} = reinterpret_cast<{self.dtype_to_cpp[dtype]}*>({ptr});"
                )
                self._argname_to_index[argname] = len(self._forms)
                self._forms.append(dtype)

            else:
                raise ak._v2._util.error(
                    TypeError(f"can't compile objects of type {type(argval)}")
                )

        eoln = "\n"
        compiler.declare(
            f"""
ssize_t {self._funcname}({", ".join(self._funcargs)}) {{
{eoln.join(self._assignments)}
{self._cppcode}

return 0;
}}
"""
        )

    def __call__(self, **kwargs):
        compiler = RawCppCompiler.instance()

        if set(kwargs) != set(self._order):
            raise ak._v2._util.error(
                TypeError(
                    f"expected arguments [{', '.join(self._order)}], not [{', '.join(kwargs)}]"
                )
            )

        # must be kept in scope while the C++ runs
        lookups = []

        arguments = []
        for argname in self._order:
            argval = kwargs[argname]
            if isinstance(argval, ak._v2.Array):
                index = self._argname_to_index[argname]

                if argval.layout.form != self._forms[index]:
                    raise ak._v2._util.error(
                        TypeError(
                            f"argument {argname} has a different Form (concrete type) from the one used in the declaration"
                        )
                    )

                lookup = ak._v2._lookup.Lookup(argval.layout)
                lookups.append(lookup)
                arguments.append(len(argval.layout))
                arguments.append(lookup.arrayptrs.ctypes.data)

            elif isinstance(argval, ak._v2.ArrayBuilder):
                index = self._argname_to_index[argname]

                if index != "ArrayBuilder":
                    raise ak._v2._util.error(
                        TypeError(
                            f"argument {argname} is an ArrayBuilder but was not declared as such"
                        )
                    )

                arguments.append(argval._layout._ptr)

            elif isinstance(argval, numpy.ndarray):
                index = self._argname_to_index[argname]

                if argval.dtype != self._forms[index]:
                    raise ak._v2._util.error(
                        TypeError(
                            f"argument {argname} has a different dtype from the one used in the declaration"
                        )
                    )

                arguments.append(argval.ctypes.data)

            else:
                raise ak._v2._util.error(
                    TypeError(
                        f"argument {argname} does not match the type used in the declaration"
                    )
                )

        # call it (no return value)
        compiler.call(self._funcname, *arguments)


class RawCppCompiler(ak.nplike.Singleton):
    # only called once; this is a Singleton
    def __init__(self):
        self._lock = threading.Lock()
        self._number = 0

        self._libAArrayclangInterpreter = ctypes.CDLL("../libAArrayclangInterpreter.so")

        self._Clang_LookupName = self._libAArrayclangInterpreter.Clang_LookupName
        self._Clang_LookupName.argtypes = [ctypes.c_char_p]
        self._Clang_LookupName.restype = ctypes.c_size_t

        self._Clang_GetFunctionAddress = (
            self._libAArrayclangInterpreter.Clang_GetFunctionAddress
        )
        self._Clang_GetFunctionAddress.argtypes = [ctypes.c_size_t]
        self._Clang_GetFunctionAddress.restype = ctypes.c_void_p

        self._Clang_Parse = self._libAArrayclangInterpreter.Clang_Parse
        self._Clang_Parse.argtypes = [ctypes.c_char_p]
        self._Clang_Parse.restype = ctypes.c_size_t

    def next_number(self):
        with self._lock:
            out = self._number
            self._number += 1
        return out

    def declare(self, cppcode):
        if self._Clang_Parse(cppcode.encode("ascii")) != 0:
            raise ak._v2._util.error(SyntaxError("invalid C++ code"))

    def call(self, name, *args):
        fn_handle = self._Clang_LookupName(name.encode("ascii"))
        fn_proto = ctypes.CFUNCTYPE(*([ctypes.c_ssize_t] * (1 + len(args))))

        fn_pointer = fn_proto(self._Clang_GetFunctionAddress(fn_handle))
        return fn_pointer(*args)

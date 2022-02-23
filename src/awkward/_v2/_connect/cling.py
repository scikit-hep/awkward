# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import base64
import struct
import json
import re

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# C++17 is required for optional, variant
headers = ("stdexcept", "string", "optional", "variant", "complex", "chrono")
cache = {}


def generate_ArrayView(compiler, use_cached=True):
    key = "ArrayView"
    if use_cached:
        out = cache.get(key)
    else:
        out = None

    if out is None:
        out = """
namespace awkward {
  class ArrayView {
  public:
    ArrayView(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : start_(start), stop_(stop), which_(which), ptrs_(ptrs) { }

    size_t size() const noexcept {{
      return stop_ - start_;
    }}

    bool empty() const noexcept {{
      return start_ == stop_;
    }}

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
        raise AssertionError(f"unrecognized Form: {type(form)}")


class Generator:
    @classmethod
    def form_from_identifier(cls, form):
        if not form.has_identifier:
            return None
        else:
            raise NotImplementedError("TODO: identifiers in C++")

    def class_type_suffix(self, key):
        return (
            base64.encodebytes(struct.pack("q", hash(key)))
            .rstrip(b"=\n")
            .replace(b"+", b"")
            .replace(b"/", b"")
            .decode("ascii")
        )

    def _generate_common(self):
        params = [
            f"if (parameter == {json.dumps(name)}) return {json.dumps(json.dumps(value))};\n      "
            for name, value in self.parameters.items()
        ]

        return f"""
    const std::string parameter(const std::string& parameter) const noexcept {{
      {"" if len(params) == 0 else "".join(x for x in params)}return "null";
    }}

    value_type at(size_t at) const {{
      if (at >= stop_ - start_) {{
        throw std::out_of_range(std::to_string(at) + " is out of range");
      }}
      else {{
        return (*this)[at];
      }}
    }}
""".strip()

    def entry(self, length="length", ptrs="ptrs", flatlist_as_rvec=False):
        key = (self, flatlist_as_rvec)
        return f"awkward::{self.class_type(key[1:])}(0, {length}, 0, {ptrs})"


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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common()}

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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef std::string value_type;

    {self._generate_common()}

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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {value_type} value_type;

    {self._generate_common()}

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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common()}

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
            raise AssertionError(index_type)
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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef std::string value_type;

    {self._generate_common()}

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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {value_type} value_type;

    {self._generate_common()}

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
      : ArrayView(start, stop, which, ptrs) {{ }}

    typedef {self.value_type(key[1:])} value_type;

    {self._generate_common()}

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
            raise AssertionError(index_type)
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

    {self._generate_common()}

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
            raise AssertionError(index_type)
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

    {self._generate_common()}

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

    {self._generate_common()}

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

    {self._generate_common()}

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

    {self._generate_common()}

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

    {self._generate_common()}

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
            raise AssertionError(index_type)
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

    {self._generate_common()}

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

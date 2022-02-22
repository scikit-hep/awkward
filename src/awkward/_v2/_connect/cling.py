# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import base64
import struct
import json

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


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
    ArrayView(ssize_t at, ssize_t which, ssize_t* ptrs)
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
        return togenerator(form.toEmptyForm(np.dtype(np.float64)))

    elif isinstance(form, ak._v2.forms.NumpyForm):
        if len(form.inner_shape) == 0:
            return NumpyGenerator.from_form(form)
        else:
            return togenerator(form.toRegularForm())

    elif isinstance(form, ak._v2.forms.RegularForm):
        return RegularGenerator.from_form(form)

    elif isinstance(form, (ak._v2.forms.ListForm, ak._v2.forms.ListOffsetForm)):
        return ListGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.IndexedForm):
        return IndexedGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.IndexedOptionForm):
        return IndexedOptionGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.ByteMaskedForm):
        return ByteMaskedGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.BitMaskedForm):
        return BitMaskedGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.UnmaskedForm):
        return UnmaskedGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.RecordForm):
        return RecordGenerator.from_form(form)

    elif isinstance(form, ak._v2.forms.UnionForm):
        return UnionGenerator.from_form(form)

    else:
        raise AssertionError(f"unrecognized Form: {type(form)}")


class Generator:
    @classmethod
    def form_from_identifier(cls, form):
        if not form.has_identifier:
            return None
        else:
            raise NotImplementedError("TODO: identifiers in C++")

    @property
    def class_type_suffix(self):
        return (
            base64.encodebytes(struct.pack("q", hash(self)))
            .rstrip(b"=\n")
            .replace(b"+", b"")
            .replace(b"/", b"")
            .decode("ascii")
        )

    def _generate_common(self):
        params = [
            f"if (parameter == {json.dumps(key)}) return {json.dumps(json.dumps(value))};\n    "
            for key, value in self.parameters.items()
        ]
        return f"""
    typedef {self.value_type} value_type;

    const std::string parameter(const std::string& parameter) const noexcept {{
      {"" if len(params) == 0 else "".join(x for x in params)}return "null";
    }}

    value_type at(size_t at) const {{
      size_t length = stop_ - start_;
      if (at >= length) {{
        throw std::out_of_range(
          std::to_string(at) + " is out of range for length " + std::to_string(length)
        );
      }}
      else {{
        return (*this)[at];
      }}
    }}
""".strip()

    def entry(self, length="length", ptrs="ptrs"):
        return f"awkward::{self.class_type}(0, {length}, 0, {ptrs})"


class NumpyGenerator(Generator, ak._v2._lookup.NumpyLookup):
    @classmethod
    def from_form(cls, form):
        return NumpyGenerator(
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
            self.primitive == other.primitive
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    @property
    def class_type(self):
        return f"NumpyArray_{self.primitive}_{self.class_type_suffix}"

    @property
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

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)

        key = (self, use_cached, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type}: public ArrayView {{
  public:
    {self.class_type}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    {self._generate_common()}

    value_type operator[](size_t at) const noexcept {{
      return reinterpret_cast<{self.value_type}*>(ptrs_[which_ + {self.ARRAY}])[start_ + at];
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class RegularGenerator(Generator, ak._v2._lookup.RegularLookup):
    @classmethod
    def from_form(cls, form):
        return RegularGenerator(
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
            self.content == other.content
            and self.size == other.size
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    @property
    def class_type(self):
        return f"RegularArray_{self.class_type_suffix}"

    @property
    def value_type(self):
        return self.content.class_type

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, use_cached, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type}: public ArrayView {{
  public:
    {self.class_type}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

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


class ListGenerator(Generator, ak._v2._lookup.ListLookup):
    @classmethod
    def from_form(cls, form):
        if isinstance(form, ak._v2.forms.ListForm):
            index_string = form.starts
        else:
            index_string = form.offsets

        return ListGenerator(
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
            self.index_type == other.index_type
            and self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

    @property
    def class_type(self):
        return f"ListArray_{self.class_type_suffix}"

    @property
    def value_type(self):
        return self.content.class_type

    def generate(self, compiler, use_cached=True, flatlist_as_rvec=False):
        generate_ArrayView(compiler, use_cached=use_cached)
        self.content.generate(compiler, use_cached, flatlist_as_rvec)

        key = (self, use_cached, flatlist_as_rvec)
        if not use_cached or key not in cache:
            out = f"""
namespace awkward {{
  class {self.class_type}: public ArrayView {{
  public:
    {self.class_type}(ssize_t start, ssize_t stop, ssize_t which, ssize_t* ptrs)
      : ArrayView(start, stop, which, ptrs) {{ }}

    {self._generate_common()}

    value_type operator[](size_t at) const noexcept {{
      ssize_t start = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STARTS}])[at];
      ssize_t stop = reinterpret_cast<{self.index_type}*>(ptrs_[which_ + {self.STOPS}])[at];
      return value_type(start, stop, ptrs_[which_ + {self.CONTENT}], ptrs_);
    }}
  }};
}}
""".strip()
            cache[key] = out
            compiler(out)


class IndexedGenerator(Generator, ak._v2._lookup.IndexedLookup):
    @classmethod
    def from_form(cls, form):
        return IndexedGenerator(
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
            self.index_type == other.index_type
            and self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )


class IndexedOptionGenerator(Generator, ak._v2._lookup.IndexedOptionLookup):
    @classmethod
    def from_form(cls, form):
        return IndexedOptionGenerator(
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
            self.index_type == other.index_type
            and self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )


class ByteMaskedGenerator(Generator, ak._v2._lookup.ByteMaskedLookup):
    @classmethod
    def from_form(cls, form):
        return ByteMaskedGenerator(
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
            self.content == other.content
            and self.valid_when == other.valid_when
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )


class BitMaskedGenerator(Generator, ak._v2._lookup.BitMaskedLookup):
    @classmethod
    def from_form(cls, form):
        return BitMaskedGenerator(
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
                self.parameters,
            )
        )

    def __eq__(self, other):
        return (
            self.content == other.content
            and self.valid_when == other.valid_when
            and self.lsb_order == other.lsb_order
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )


class UnmaskedGenerator(Generator, ak._v2._lookup.UnmaskedLookup):
    @classmethod
    def from_form(cls, form):
        return UnmaskedGenerator(
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
            self.content == other.content
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )


class RecordGenerator(Generator, ak._v2._lookup.RecordLookup):
    @classmethod
    def from_form(cls, form):
        return RecordGenerator(
            [togenerator(x) for x in form.contents],
            None if form.is_tuple else form.fields,
            cls.form_from_identifier(form),
            form.parameters,
        )

    def __init__(self, contents, fields, identifier, parameters):
        self.contents = contents
        self.fields = fields
        self.identifier = identifier
        self.parameters = parameters

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
            self.contents == other.contents
            and self.fields == other.fields
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )


class UnionGenerator(Generator, ak._v2._lookup.UnionLookup):
    @classmethod
    def from_form(cls, form):
        return UnionGenerator(
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
        self.contents = contents
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
            self.index_type == other.index_type
            and self.contents == other.contents
            and self.identifier == other.identifier
            and self.parameters == other.parameters
        )

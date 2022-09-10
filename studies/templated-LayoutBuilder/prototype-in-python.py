import json

import pytest

import numpy as np
import awkward as ak


############################### GrowableBuffer


class Ref:
    """
    Python can't pass some types by reference, so I'll do this to emulate 'int&' in C.
    """

    def __init__(self, value):
        self.value = value


class GenericRef(Ref):
    """
    Oh, the things you can do in C++!
    """

    def __init__(self, get, set):
        self.get = get
        self.set = set

    @property
    def value(self):
        return self.get()

    @value.setter
    def value(self, new_value):
        return self.set(new_value)


class GrowableBuffer:
    """
    This is a MOCK GrowableBuffer, just so that I have something to use below.
    """

    def __init__(self, PRIMITIVE):
        self.fake = []
        self.PRIMITIVE = PRIMITIVE

    def append(self, x):
        self.fake.append(x)

    def extend(self, xs, size):
        self.fake.extend(xs)

    def clear(self):
        self.fake.clear()

    def length(self):
        return len(self.fake)

    def last(self):
        # raise an error if length == 0
        return self.fake[-1]

    def append_and_get_ref(self, x):
        """
        Like append, but the type signature returns '&PRIMITIVE'.

        ```c++
        template <typename PRIMITIVE>
        &PRIMITIVE GrowableBuffer<PRIMITIVE>::append_and_get_ref(PRIMITIVE datum) {
            append(datum);
            return (&*ptr_.back())[length_.back()];
        }
        ```
        """
        index = len(self.fake)
        self.fake.append(x)

        def get():
            return self.fake[index]

        def set(new_value):
            self.fake[index] = new_value

        return GenericRef(get, set)

    def nbytes(self):
        return self.length() * np.dtype(self.PRIMITIVE).itemsize

    def concatenate(self, external_pointer):
        external_pointer.view(self.PRIMITIVE)[:] = self.fake

    def index_form(self):
        if self.PRIMITIVE == "uint8":
            return "u8"
        elif self.PRIMITIVE == "int8":
            return "i8"
        elif self.PRIMITIVE == "int32":
            return "i32"
        elif self.PRIMITIVE == "uint32":
            return "u32"
        elif self.PRIMITIVE == "int64":
            return "i64"
        else:
            raise AssertionError


# What are the methods that all BUILDERs have?
# When using templates, they're not enforced by class inheritance.
# But we can specify them for ourselves:
#
#     * std::string parameters() noexcept const
#
#     * void set_id(size_t id) noexcept   // to give each node a unique id
#
#     * void clear() noexcept   // I don't think this is necessary, but okay...
#
#     * size_t length() noexcept const
#
#     * bool is_valid(std::string &error) noexcept const   // this checks for consistency
#
#     * void buffer_nbytes(std::map<std::string, size_t> &names_nbytes) noexcept const
#
#     * void to_buffers(std::map<std::string, void*> &buffers) noexcept const
#
#     * std::string form() noexcept const


############################### NumpyForm


class Numpy:
    def __init__(self, PRIMITIVE, parameters):
        self.data_ = GrowableBuffer(PRIMITIVE)
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def append(self, x):
        self.data_.append(x)

    def extend(self, xs, size):
        self.data_.extend(xs, size)

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1

    def clear(self):
        self.data_.clear()

    def length(self):
        return self.data_.length()

    def is_valid(self, error: Ref(str)):
        return True

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-data"] = self.data_.nbytes()

    def to_buffers(self, buffers):
        self.data_.concatenate(buffers[f"node{self.id_}-data"])

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "NumpyArray", "primitive": {json.dumps(self.data_.PRIMITIVE)}, "form_key": "node{self.id_}"{params}}}'


############################### EmptyForm


class Empty:
    def __init__(self, parameters):
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1

    def clear(self):
        pass

    def length(self):
        return 0

    def is_valid(self, error: Ref(str)):
        return True

    def buffer_nbytes(self, names_nbytes):
        pass

    def to_buffers(self, buffers):
        pass

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "EmptyArray"{params}}}'


############################### ListOffsetForm


class ListOffset:
    def __init__(self, PRIMITIVE, content, parameters):
        self.offsets_ = GrowableBuffer(PRIMITIVE)
        self.offsets_.append(0)
        self.content_ = content
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def begin_list(self):
        return self.content_

    def end_list(self):
        self.offsets_.append(self.content_.length())

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.offsets_.clear()
        self.offsets_.append(0)
        self.content_.clear()

    def length(self):
        return self.offsets_.length() - 1

    def is_valid(self, error: Ref(str)):
        if self.content_.length() != self.offsets_.last():
            error.value = f"ListOffset node{self.id_} has content length {self.content_.length()} but last offset {self.offsets_.last()}"
            return False
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-offsets"] = self.offsets_.nbytes()
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.offsets_.concatenate(buffers[f"node{self.id_}-offsets"])
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "ListOffsetArray", "offsets": "{self.offsets_.index_form()}", "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### ListForm


class List:
    def __init__(self, PRIMITIVE, content, parameters):
        self.starts_ = GrowableBuffer(PRIMITIVE)
        self.stops_ = GrowableBuffer(PRIMITIVE)
        self.content_ = content
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def begin_list(self):
        self.starts_.append(self.content_.length())
        return self.content_

    def end_list(self):
        self.stops_.append(self.content_.length())

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.starts_.clear()
        self.stops_.clear()
        self.content_.clear()

    def length(self):
        return self.starts_.length()

    def is_valid(self, error: Ref(str)):
        if self.starts_.length() != self.stops_.length():
            error.value = f"List node{self.id_} has starts length {self.starts_.length()} but stops length {self.stops_.length()}"
        elif self.stops_.length() > 0 and self.content_.length() != self.stops_.last():
            error.value = f"List node{self.id_} has content length {self.content_.length()} but last stops {self.stops_.last()}"
            return False
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-starts"] = self.starts_.nbytes()
        names_nbytes[f"node{self.id_}-stops"] = self.stops_.nbytes()
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.starts_.concatenate(buffers[f"node{self.id_}-starts"])
        self.stops_.concatenate(buffers[f"node{self.id_}-stops"])
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "ListArray", "starts": "{self.starts_.index_form()}", "stops": "{self.stops_.index_form()}", "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### RegularForm


class Regular:
    def __init__(self, content, size, parameters):
        self.length_ = 0
        self.content_ = content
        self.size_ = size
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def begin_list(self):
        return self.content_

    def end_list(self):
        self.length_ += 1

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.content_.clear()

    def length(self):
        return self.length_

    def is_valid(self, error: Ref(str)):
        if self.content_.length() != self.length_ * self.size_:
            error.value = f"Regular node{self.id_} has content length {self.content_.length()}, but length {self.length_} and size {self.size_}"
            return False
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "RegularArray", "size": {self.size_}, "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### IndexedForm


class Indexed:
    def __init__(self, PRIMITIVE, content, parameters):
        self.last_valid_ = -1
        self.index_ = GrowableBuffer(PRIMITIVE)
        self.content_ = content
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def append_index(self):
        self.last_valid_ = self.content_.length()
        self.index_.append(self.last_valid_)
        return self.content_

    def extend_index(self, size):
        start = self.content_.length()
        stop = start + size
        self.last_valid_ = stop - 1
        self.index_.extend(list(range(start, stop)), size)
        return self.content_

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.last_valid_ = -1
        self.index_.clear()
        self.content_.clear()

    def length(self):
        return self.index_.length()

    def is_valid(self, error: Ref(str)):
        if self.content_.length() != self.index_.length():
            error.value = f"Indexed node{self.id_} has content length {self.content_.length()} but index length {self.index_.length()}"
            return False
        elif self.content_.length() != self.last_valid_ + 1:
            error.value = f"Indexed node{self.id_} has content length {self.content_.length()} but last valid index is {self.last_valid_}"
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-index"] = self.index_.nbytes()
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.index_.concatenate(buffers[f"node{self.id_}-index"])
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "IndexedArray", "index": "{self.index_.index_form()}", "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### IndexedOptionForm


class IndexedOption:
    def __init__(self, PRIMITIVE, content, parameters):
        self.last_valid_ = -1
        self.index_ = GrowableBuffer(PRIMITIVE)
        self.content_ = content
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def append_index(self):
        self.last_valid_ = self.content_.length()
        self.index_.append(self.last_valid_)
        return self.content_

    def extend_index(self, size):
        start = self.content_.length()
        stop = start + size
        self.last_valid_ = stop - 1
        self.index_.extend(list(range(start, stop)), size)
        return self.content_

    def append_null(self):
        self.index_.append(-1)

    def extend_null(self, size):
        self.index_.extend([-1] * size, size)

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.last_valid_ = -1
        self.index_.clear()
        self.content_.clear()

    def length(self):
        return self.index_.length()

    def is_valid(self, error: Ref(str)):
        if self.content_.length() != self.last_valid_ + 1:
            error.value = f"Indexed node{self.id_} has content length {self.content_.length()} but last valid index is {self.last_valid_}"
            return False
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-index"] = self.index_.nbytes()
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.index_.concatenate(buffers[f"node{self.id_}-index"])
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "IndexedOptionArray", "index": "{self.index_.index_form()}", "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### ByteMaskedForm


class ByteMasked:
    def __init__(self, content, valid_when, parameters):
        self.mask_ = GrowableBuffer("int8")
        self.content_ = content
        self.valid_when_ = valid_when
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def valid_when(self):
        return self.valid_when_

    def append_valid(self):
        self.mask_.append(self.valid_when_)
        return self.content_

    def extend_valid(self, size):
        self.mask_.extend([self.valid_when_] * size, size)
        return self.content_

    def append_null(self):
        self.mask_.append(not self.valid_when_)
        return self.content_

    def extend_null(self, size):
        self.mask_.extend([not self.valid_when_] * size, size)
        return self.content_

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.mask_.clear()
        self.content_.clear()

    def length(self):
        return self.mask_.length()

    def is_valid(self, error: Ref(str)):
        if self.content_.length() != self.mask_.length():
            error.value = f"ByteMasked node{self.id_} has content length {self.content_.length()} but mask length {self.stops_.length()}"
            return False
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-mask"] = self.mask_.nbytes()
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.mask_.concatenate(buffers[f"node{self.id_}-mask"])
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "ByteMaskedArray", "mask": "{self.mask_.index_form()}", "valid_when": {json.dumps(self.valid_when_)}, "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### BitMaskedForm


class BitMasked:
    def __init__(self, content, valid_when, lsb_order, parameters):
        self.mask_ = GrowableBuffer("uint8")
        self.content_ = content
        self.valid_when_ = valid_when
        self.lsb_order_ = lsb_order
        self.current_byte_ = np.uint8(0)
        self.current_byte_ref_ = self.mask_.append_and_get_ref(self.current_byte_)
        self.current_index_ = 0
        if self.lsb_order_:
            self.cast_ = np.array(
                [
                    np.uint8(1 << 0),
                    np.uint8(1 << 1),
                    np.uint8(1 << 2),
                    np.uint8(1 << 3),
                    np.uint8(1 << 4),
                    np.uint8(1 << 5),
                    np.uint8(1 << 6),
                    np.uint8(1 << 7),
                ]
            )
        else:
            self.cast_ = np.array(
                [
                    np.uint8(128 >> 0),
                    np.uint8(128 >> 1),
                    np.uint8(128 >> 2),
                    np.uint8(128 >> 3),
                    np.uint8(128 >> 4),
                    np.uint8(128 >> 5),
                    np.uint8(128 >> 6),
                    np.uint8(128 >> 7),
                ]
            )
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def valid_when(self):
        return self.valid_when_

    def lsb_order(self):
        return self.lsb_order_

    def _append_begin(self):
        """
        Private helper function.
        """
        if self.current_index_ == 8:
            self.current_byte_ = np.uint8(0)
            self.current_byte_ref_ = self.mask_.append_and_get_ref(self.current_byte_)
            self.current_index_ = 0

    def _append_end(self):
        """
        Private helper function.
        """
        self.current_index_ += 1
        if self.valid_when_:
            # 0 indicates null, 1 indicates valid
            self.current_byte_ref_.value = self.current_byte_
        else:
            # 0 indicates valid, 1 indicates null
            self.current_byte_ref_.value = ~self.current_byte_

    def append_valid(self):
        self._append_begin()
        # current_byte_ and cast_: 0 indicates null, 1 indicates valid
        self.current_byte_ |= self.cast_[self.current_index_]
        self._append_end()
        return self.content_

    def extend_valid(self, size):
        # Just an interface; not actually faster than calling append many times.
        for _ in range(size):
            self.append_valid()
        return self.content_

    def append_null(self):
        self._append_begin()
        # current_byte_ and cast_ default to null, no change
        self._append_end()
        return self.content_

    def extend_null(self, size):
        # Just an interface; not actually faster than calling append many times.
        for _ in range(size):
            self.append_null()
        return self.content_

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.mask_.clear()
        self.content_.clear()

    def length(self):
        return (self.mask_.length() - 1) * 8 + self.current_index_

    def is_valid(self, error: Ref(str)):
        if self.content_.length() != self.length():
            error.value = f"BitMasked node{self.id_} has content length {self.content_.length()} but bit mask length {self.length()}"
            return False
        else:
            return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-mask"] = self.mask_.nbytes()
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.mask_.concatenate(buffers[f"node{self.id_}-mask"])
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "BitMaskedArray", "mask": "{self.mask_.index_form()}", "valid_when": {json.dumps(self.valid_when_)}, "lsb_order": {json.dumps(self.lsb_order_)}, "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### UnmaskedForm


class Unmasked:
    def __init__(self, content, parameters):
        self.content_ = content
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def append_valid(self):
        return self.content_

    def extend_valid(self, size):
        return self.content_

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.content_.clear()

    def length(self):
        return self.content_.length()

    def is_valid(self, error: Ref(str)):
        return self.content_.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        self.content_.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.content_.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        return f'{{"class": "UnmaskedArray", "content": {self.content_.form()}, "form_key": "node{self.id_}"{params}}}'


############################### RecordForm


class FieldPair:
    """
    I'm assuming that C++ templates can't be variadic without the repeated part
    being a single type; hence, this Field class.
    """

    def __init__(self, name, content):
        self.name = name
        self.content = content


class Record:
    def __init__(self, field_pairs, parameters):
        assert len(field_pairs) != 0
        # C++ can actually do lookup by the compile-time constant field name.
        # There really isn't an equivalent of that in Python. Pretend this is that.
        self.field_pairs_ = {pair.name: pair for pair in field_pairs}
        self.first_content_ = field_pairs[0].content
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def field(self, name):
        return self.field_pairs_[name].content

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        for pair in self.field_pairs_.values():
            pair.content.set_id(id)

    def clear(self):
        for pair in self.field_pairs_.values():
            pair.content.clear()

    def length(self):
        return self.first_content_.length()

    def is_valid(self, error: Ref(str)):
        length = -1
        for pair in self.field_pairs_.values():
            if length == -1:
                length = pair.content.length()
            elif length != pair.content.length():
                error.value = f"Record node{self.id_} has field {pair.name} length {pair.content.length()} that differs from the first length {length}"
                return False
        for pair in self.field_pairs_.values():
            if not pair.content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        for pair in self.field_pairs_.values():
            pair.content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        for pair in self.field_pairs_.values():
            pair.content.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        pairs = ", ".join(
            f"{json.dumps(pair.name)}: {pair.content.form()}"
            for pair in self.field_pairs_.values()
        )
        return f'{{"class": "RecordArray", "contents": {{{pairs}}}, "form_key": "node{self.id_}"{params}}}'


class Tuple:
    def __init__(self, contents, parameters):
        assert len(contents) != 0
        self.contents_ = contents
        self.first_content_ = contents[0]
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def index(self, at):
        return self.contents_[at]

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        for content in self.contents_:
            content.set_id(id)

    def clear(self):
        for content in self.contents_:
            pair.content.clear()

    def length(self):
        return self.first_content_.length()

    def is_valid(self, error: Ref(str)):
        length = -1
        for index, content in enumerate(self.contents_):
            if length == -1:
                length = content.length()
            elif length != content.length():
                error.value = f"Tuple node{self.id_} has index {index} length {content.length()} that differs from the first length {length}"
                return False
        for content in self.contents_:
            if not content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        for content in self.contents_:
            content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        for content in self.contents_:
            content.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        contents = ", ".join(content.form() for content in self.contents_)
        return f'{{"class": "RecordArray", "contents": [{contents}], "form_key": "node{self.id_}"{params}}}'


class EmptyRecord:
    def __init__(self, is_tuple, parameters):
        self.length_ = 0
        self.is_tuple_ = is_tuple
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def append(self):
        self.length_ += 1

    def extend(self, size):
        self.length_ += size

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value

    def clear(self):
        self.length_ = 0

    def length(self):
        return self.length_

    def is_valid(self, error: Ref(str)):
        return True

    def buffer_nbytes(self, names_nbytes):
        pass

    def to_buffers(self, buffers):
        pass

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        if self.is_tuple_:
            return f'{{"class": "RecordArray", "contents": [], "form_key": "node{self.id_}"{params}}}'
        else:
            return f'{{"class": "RecordArray", "contents": {{}}, "form_key": "node{self.id_}"{params}}}'


############################### UnionForm


class Union:
    def __init__(self, PRIMITIVE, contents, parameters):
        self.last_valid_index_ = [-1] * len(contents)
        self.tags_ = GrowableBuffer("int8")
        self.index_ = GrowableBuffer(PRIMITIVE)
        self.contents_ = contents
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def append_index(self, tag):
        which_content = self.contents_[tag]
        next_index = which_content.length()
        self.last_valid_index_[tag] = next_index
        self.tags_.append(tag)
        self.index_.append(next_index)
        return which_content

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        for content in self.contents_:
            content.set_id(id)

    def clear(self):
        for tag in range(len(self.last_valid_index_)):
            self.last_valid_index_[tag] = -1
        self.tags_.clear()
        self.index_.clear()
        for content in self.contents_:
            content.clear()

    def length(self):
        return self.tags_.length()

    def is_valid(self, error: Ref(str)):
        for tag in range(len(self.last_valid_index_)):
            if self.contents_[tag].length() != self.last_valid_index_[tag] + 1:
                error.value = f"Union node{self.id_} has content {tag} length {self.contents_[tag].length()} but last valid index is {self.last_valid_index_[tag]}"
                return False
        for content in self.contents_:
            if not content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self.id_}-tags"] = self.tags_.nbytes()
        names_nbytes[f"node{self.id_}-index"] = self.index_.nbytes()
        for content in self.contents_:
            content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.tags_.concatenate(buffers[f"node{self.id_}-tags"])
        self.index_.concatenate(buffers[f"node{self.id_}-index"])
        for content in self.contents_:
            content.to_buffers(buffers)

    def form(self):
        params = "" if self.parameters_ == "" else f", parameters: {self.parameters_}"
        contents = ", ".join(content.form() for content in self.contents_)
        return f'{{"class": "UnionArray", "tags": "{self.tags_.index_form()}", "index": "{self.index_.index_form()}", "contents": [{contents}], "form_key": "node{self.id_}"{params}}}'


############################### tests


def test_Numpy():
    builder = Numpy("float64", "")

    builder.append(1.1)
    builder.append(2.2)
    builder.extend([3.3, 4.4, 5.5], 3)

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 1

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node0-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "NumpyArray", "primitive": "float64", "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_Empty():
    builder = Empty("")

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 0

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)

    assert builder.form() == '{"class": "EmptyArray"}'

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == []


def test_ListOffset():
    builder = ListOffset("int64", Numpy("float64", ""), "")

    subbuilder = builder.begin_list()
    subbuilder.append(1.1)
    subbuilder.append(2.2)
    subbuilder.append(3.3)
    builder.end_list()

    subbuilder = builder.begin_list()
    builder.end_list()

    subbuilder = builder.begin_list()
    subbuilder.append(4.4)
    subbuilder.append(5.5)
    builder.end_list()

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 2

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node0-offsets"].view("int64").tolist() == [0, 3, 3, 5]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "ListOffsetArray", "offsets": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_List():
    builder = List("int64", Numpy("float64", ""), "")

    subbuilder = builder.begin_list()
    subbuilder.append(1.1)
    subbuilder.append(2.2)
    subbuilder.append(3.3)
    builder.end_list()

    subbuilder = builder.begin_list()
    builder.end_list()

    subbuilder = builder.begin_list()
    subbuilder.append(4.4)
    subbuilder.append(5.5)
    builder.end_list()

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 3

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node0-starts"].view("int64").tolist() == [0, 3, 3]
    assert buffers["node0-stops"].view("int64").tolist() == [3, 3, 5]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "ListArray", "starts": "i64", "stops": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_Regular():
    builder = Regular(Numpy("float64", ""), 3, "")

    subbuilder = builder.begin_list()
    subbuilder.append(1.1)
    subbuilder.append(2.2)
    subbuilder.append(3.3)
    builder.end_list()

    subbuilder = builder.begin_list()
    subbuilder.append(4.4)
    subbuilder.append(5.5)
    subbuilder.append(6.6)
    builder.end_list()

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 1

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node1-data"].view("float64").tolist() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
    ]

    assert (
        builder.form()
        == '{"class": "RegularArray", "size": 3, "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]


def test_Regular_size0():
    builder = Regular(Numpy("float64", ""), 0, "")

    subbuilder = builder.begin_list()
    builder.end_list()

    subbuilder = builder.begin_list()
    builder.end_list()

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 1

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node1-data"].view("float64").tolist() == []

    assert (
        builder.form()
        == '{"class": "RegularArray", "size": 0, "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [[], []]


def test_Indexed():
    builder = Indexed("int64", Numpy("float64", ""), "")

    subbuilder = builder.append_index()
    subbuilder.append(1.1)

    subbuilder = builder.append_index()
    subbuilder.append(2.2)

    subbuilder = builder.extend_index(3)
    subbuilder.extend([3.3, 4.4, 5.5], 3)

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 2

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node0-index"].view("int64").tolist() == [0, 1, 2, 3, 4]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "IndexedArray", "index": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_IndexedOption():
    builder = IndexedOption("int64", Numpy("float64", ""), "")

    subbuilder = builder.append_index()
    subbuilder.append(1.1)

    builder.append_null()

    subbuilder = builder.extend_index(3)
    subbuilder.extend([3.3, 4.4, 5.5], 3)

    builder.extend_null(2)

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 2

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node0-index"].view("int64").tolist() == [0, -1, 1, 2, 3, -1, -1]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "IndexedOptionArray", "index": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, None, 3.3, 4.4, 5.5, None, None]


@pytest.mark.parametrize("valid_when", [False, True])
def test_ByteMasked(valid_when):
    builder = ByteMasked(Numpy("float64", ""), valid_when, "")

    subbuilder = builder.append_valid()
    subbuilder.append(1.1)

    subbuilder = builder.append_null()
    subbuilder.append(-1000)  # have to supply a "dummy" value

    subbuilder = builder.extend_valid(3)
    subbuilder.extend([3.3, 4.4, 5.5], 3)

    subbuilder = builder.extend_null(2)
    for _ in range(2):
        subbuilder.append(-1000)  # have to supply a "dummy" value

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 2

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    if valid_when:
        assert buffers["node0-mask"].view("int8").tolist() == [1, 0, 1, 1, 1, 0, 0]
    else:
        assert buffers["node0-mask"].view("int8").tolist() == [0, 1, 0, 0, 0, 1, 1]
    assert buffers["node1-data"].view("float64").tolist() == [
        1.1,
        -1000,
        3.3,
        4.4,
        5.5,
        -1000,
        -1000,
    ]

    assert (
        builder.form()
        == f'{{"class": "ByteMaskedArray", "mask": "i8", "valid_when": {json.dumps(valid_when)}, "content": {{"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}}, "form_key": "node0"}}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, None, 3.3, 4.4, 5.5, None, None]


@pytest.mark.parametrize("valid_when", [False, True])
@pytest.mark.parametrize("lsb_order", [False, True])
def test_BitMasked(valid_when, lsb_order):
    builder = BitMasked(Numpy("float64", ""), valid_when, lsb_order, "")

    error = Ref("")
    assert builder.is_valid(error), error.value

    subbuilder = builder.append_valid()
    subbuilder.append(1.1)

    assert builder.is_valid(error), error.value

    subbuilder = builder.append_null()
    subbuilder.append(-1000)  # have to supply a "dummy" value

    assert builder.is_valid(error), error.value

    subbuilder = builder.extend_valid(3)
    subbuilder.extend([3.3, 4.4, 5.5], 3)

    assert builder.is_valid(error), error.value

    subbuilder = builder.extend_null(2)
    for _ in range(2):
        subbuilder.append(-1000)  # have to supply a "dummy" value

    assert builder.is_valid(error), error.value

    assert builder.is_valid(error), error.value

    subbuilder = builder.append_valid()
    subbuilder.append(8)
    subbuilder = builder.append_valid()
    subbuilder.append(9)
    subbuilder = builder.append_valid()
    subbuilder.append(10)

    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 2

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    if valid_when and lsb_order:
        assert buffers["node0-mask"].view("uint8").tolist() == [157, 3]
    elif valid_when and not lsb_order:
        assert buffers["node0-mask"].view("uint8").tolist() == [185, 192]
    elif not valid_when and lsb_order:
        assert buffers["node0-mask"].view("uint8").tolist() == [98, 252]
    elif not valid_when and not lsb_order:
        assert buffers["node0-mask"].view("uint8").tolist() == [70, 63]
    assert buffers["node1-data"].view("float64").tolist() == [
        1.1,
        -1000,
        3.3,
        4.4,
        5.5,
        -1000,
        -1000,
        8,
        9,
        10,
    ]

    assert (
        builder.form()
        == f'{{"class": "BitMaskedArray", "mask": "u8", "valid_when": {json.dumps(valid_when)}, "lsb_order": {json.dumps(lsb_order)}, "content": {{"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}}, "form_key": "node0"}}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, None, 3.3, 4.4, 5.5, None, None, 8, 9, 10]


def test_Unmasked():
    builder = Unmasked(Numpy("float64", ""), "")

    subbuilder = builder.append_valid()
    subbuilder.append(1.1)

    subbuilder = builder.append_valid()
    subbuilder.append(2.2)

    subbuilder = builder.extend_valid(3)
    subbuilder.extend([3.3, 4.4, 5.5], 3)

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 1

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        builder.form()
        == '{"class": "UnmaskedArray", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_Record_Numpy_ListOffset():
    builder = Record(
        [
            FieldPair("one", Numpy("float64", "")),
            FieldPair("two", ListOffset("int64", Numpy("int32", ""), "")),
        ],
        "",
    )

    error = Ref("")
    assert builder.is_valid(error), error.value

    subbuilder_one = builder.field("one")
    subbuilder_one.append(1.1)
    subbuilder_two = builder.field("two")
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    subbuilder_one = builder.field("one")
    subbuilder_one.append(2.2)
    subbuilder_two = builder.field("two")
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subsubbuilder.append(2)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    subbuilder_one = builder.field("one")
    subbuilder_one.append(3.3)
    subbuilder_two = builder.field("two")
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subsubbuilder.append(2)
    subsubbuilder.append(3)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 3

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3]
    assert buffers["node2-offsets"].view("int64").tolist() == [0, 1, 3, 6]
    assert buffers["node3-data"].view("int32").tolist() == [1, 1, 2, 1, 2, 3]

    assert (
        builder.form()
        == '{"class": "RecordArray", "contents": {"one": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "two": {"class": "ListOffsetArray", "offsets": "i64", "content": {"class": "NumpyArray", "primitive": "int32", "form_key": "node3"}, "form_key": "node2"}}, "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [
        {"one": 1.1, "two": [1]},
        {"one": 2.2, "two": [1, 2]},
        {"one": 3.3, "two": [1, 2, 3]},
    ]


def test_Tuple_Numpy_ListOffset():
    builder = Tuple(
        [Numpy("float64", ""), ListOffset("int64", Numpy("int32", ""), "")], ""
    )

    error = Ref("")
    assert builder.is_valid(error), error.value

    subbuilder_one = builder.index(0)
    subbuilder_one.append(1.1)
    subbuilder_two = builder.index(1)
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    subbuilder_one = builder.index(0)
    subbuilder_one.append(2.2)
    subbuilder_two = builder.index(1)
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subsubbuilder.append(2)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    subbuilder_one = builder.index(0)
    subbuilder_one.append(3.3)
    subbuilder_two = builder.index(1)
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subsubbuilder.append(2)
    subsubbuilder.append(3)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 3

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3]
    assert buffers["node2-offsets"].view("int64").tolist() == [0, 1, 3, 6]
    assert buffers["node3-data"].view("int32").tolist() == [1, 1, 2, 1, 2, 3]

    assert (
        builder.form()
        == '{"class": "RecordArray", "contents": [{"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, {"class": "ListOffsetArray", "offsets": "i64", "content": {"class": "NumpyArray", "primitive": "int32", "form_key": "node3"}, "form_key": "node2"}], "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]


@pytest.mark.parametrize("is_tuple", [False, True])
def test_EmptyRecord(is_tuple):
    builder = EmptyRecord(is_tuple, "")

    error = Ref("")
    assert builder.is_valid(error), error.value

    builder.append()

    assert builder.is_valid(error), error.value

    builder.extend(2)

    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 0

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)

    if is_tuple:
        assert (
            builder.form()
            == '{"class": "RecordArray", "contents": [], "form_key": "node0"}'
        )
    else:
        assert (
            builder.form()
            == '{"class": "RecordArray", "contents": {}, "form_key": "node0"}'
        )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    if is_tuple:
        assert array.tolist() == [(), (), ()]
    else:
        assert array.tolist() == [{}, {}, {}]


def test_Union_Numpy_ListOffset():
    builder = Union(
        "uint32",
        [Numpy("float64", ""), ListOffset("int64", Numpy("int32", ""), "")],
        "",
    )

    error = Ref("")
    assert builder.is_valid(error), error.value

    subbuilder_one = builder.append_index(0)
    subbuilder_one.append(1.1)

    assert builder.is_valid(error), error.value

    subbuilder_two = builder.append_index(1)
    subsubbuilder = subbuilder_two.begin_list()
    subsubbuilder.append(1)
    subsubbuilder.append(2)
    subbuilder_two.end_list()

    assert builder.is_valid(error), error.value

    subbuilder_one = builder.append_index(0)
    subbuilder_one.append(3.3)

    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 5

    buffers = {
        name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()
    }
    builder.to_buffers(buffers)
    assert buffers["node0-tags"].view("int8").tolist() == [0, 1, 0]
    assert buffers["node0-index"].view("uint32").tolist() == [0, 0, 1]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 3.3]
    assert buffers["node2-offsets"].view("int64").tolist() == [0, 2]
    assert buffers["node3-data"].view("int32").tolist() == [1, 2]

    assert (
        builder.form()
        == '{"class": "UnionArray", "tags": "i8", "index": "u32", "contents": [{"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, {"class": "ListOffsetArray", "offsets": "i64", "content": {"class": "NumpyArray", "primitive": "int32", "form_key": "node3"}, "form_key": "node2"}], "form_key": "node0"}'
    )

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, [1, 2], 3.3]

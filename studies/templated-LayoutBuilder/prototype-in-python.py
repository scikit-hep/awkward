import json

import numpy as np
import awkward._v2 as ak


############################### GrowableBuffer

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
        # throws an error if length == 0
        return self.fake[-1]

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


class Ref:
    """
    Python can't pass some types by reference, so I'll do this to emulate 'int&' in C.
    """
    def __init__(self, value):
        self.value = value


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
            return True

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
            return True

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
        self.content_ = content
        self.size_ = size
        self.parameters_ = parameters
        self.set_id(Ref(0))

    def content(self):
        return self.content_

    def begin_list(self):
        return self.content_

    def end_list(self):
        pass

    def parameters(self):
        return self.parameters_

    def set_id(self, id: Ref(int)):
        self.id_ = id.value
        id.value += 1
        self.content_.set_id(id)

    def clear(self):
        self.content_.clear()

    def length(self):
        return self.content_.length() // self.size_

    def is_valid(self, error: Ref(str)):
        if self.content_.length() % self.size_ != 0:
            error.value = f"Regular node{self.id_} has content length {self.content_.length()} but size {self.size_}"
            return False
        else:
            return True

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
            return True

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
            return True

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
            return True

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
############################### UnmaskedForm
############################### RecordForm
############################### UnionForm


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

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node0-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert builder.form() == '{"class": "NumpyArray", "primitive": "float64", "form_key": "node0"}'

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_Empty():
    builder = Empty("")

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 0

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
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

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node0-offsets"].view("int64").tolist() == [0, 3, 3, 5]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert builder.form() == '{"class": "ListOffsetArray", "offsets": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'

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

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node0-starts"].view("int64").tolist() == [0, 3, 3]
    assert buffers["node0-stops"].view("int64").tolist() == [3, 3, 5]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert builder.form() == '{"class": "ListArray", "starts": "i64", "stops": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'

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

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    assert builder.form() == '{"class": "RegularArray", "size": 3, "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]


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

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node0-index"].view("int64").tolist() == [0, 1, 2, 3, 4]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert builder.form() == '{"class": "IndexedArray", "index": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'

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

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node0-index"].view("int64").tolist() == [0, -1, 1, 2, 3, -1, -1]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, 3.3, 4.4, 5.5]

    assert builder.form() == '{"class": "IndexedOptionArray", "index": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, None, 3.3, 4.4, 5.5, None, None]


def test_ByteMasked():
    builder = ByteMasked(Numpy("float64", ""), False, "")

    subbuilder = builder.append_valid()
    subbuilder.append(1.1)

    subbuilder = builder.append_null()
    subbuilder.append(-1000)   # have to supply a "dummy" value

    subbuilder = builder.extend_valid(3)
    subbuilder.extend([3.3, 4.4, 5.5], 3)

    subbuilder = builder.extend_null(2)
    for _ in range(2):
        subbuilder.append(-1000)   # have to supply a "dummy" value

    error = Ref("")
    assert builder.is_valid(error), error.value

    names_nbytes = {}
    builder.buffer_nbytes(names_nbytes)
    assert len(names_nbytes) == 2

    buffers = {name: np.empty(nbytes, np.uint8) for name, nbytes in names_nbytes.items()}
    builder.to_buffers(buffers)
    assert buffers["node0-mask"].view("int8").tolist() == [0, 1, 0, 0, 0, 1, 1]
    assert buffers["node1-data"].view("float64").tolist() == [1.1, -1000, 3.3, 4.4, 5.5, -1000, -1000]

    assert builder.form() == '{"class": "ByteMaskedArray", "mask": "i8", "valid_when": false, "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "node1"}, "form_key": "node0"}'

    array = ak.from_buffers(builder.form(), builder.length(), buffers)
    assert array.tolist() == [1.1, None, 3.3, 4.4, 5.5, None, None]

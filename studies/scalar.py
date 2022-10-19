import jax.numpy

import awkward as ak

import numbers


# Can't subclass jax.numpy.DeviceArray
# See https://github.com/google/jax/issues/4269#issuecomment-691402423
class JaxScalar:
    def __init__(self, scalar):
        self._value = scalar

    @property
    def value(self):
        return self._value

    def __getitem__(self, item):
        return self._value.__getitem__(item)

    def __item__(self):
        return self._value.__item__()

    def __str__(self):
        return "JAXSCALAR " + str(self._value)

    def __repr__(self):
        return "JAXSCALAR " + repr(self._value)

    def __format__(self, format_spec):
        return "JAXSCALAR " + format(self._value, format_spec)

    def __bytes__(self):
        return bytes(self._value)

    def _compops(self, op, other):
        if isinstance(other, JaxScalar):
            return op(other._value)
        elif isinstance(other, numbers.Number):
            return op(other)
        elif hasattr(other, "shape") and other.shape == ():
            return op(other)
        else:
            return NotImplemented

    def __lt__(self, other):
        return self._compops(self._value.__lt__, other)

    def __le__(self, other):
        return self._compops(self._value.__le__, other)

    def __gt__(self, other):
        return self._compops(self._value.__gt__, other)

    def __ge__(self, other):
        return self._compops(self._value.__ge__, other)

    def __eq__(self, other):
        return self._compops(self._value.__eq__, other)

    def __ne__(self, other):
        return self._compops(self._value.__ne__, other)

    def __hash__(self):
        return hash(self._value)

    def __bool__(self):
        return bool(self._value)

    __slots__ = ["_value"]

    def __len__(self):
        raise ak._errors.wrap_error(TypeError("len() of unsized object"))

    def _binop(self, op, other):
        if isinstance(other, JaxScalar):
            return JaxScalar(op(other._value))
        elif isinstance(other, numbers.Number):
            return JaxScalar(op(other))
        elif hasattr(other, "shape") and other.shape == ():
            return JaxScalar(
                getattr(other, op.__name__.replace("__", "__r", 1))(self._value)
            )
        else:
            return NotImplemented

    def __array__(self):
        import numpy

        return numpy.array(self._value)

    def __add__(self, other):
        return self._binop(self._value.__add__, other)

    def __sub__(self, other):
        return self._binop(self._value.__sub__, other)

    def __mul__(self, other):
        return self._binop(self._value.__mul__, other)

    def __matmul__(self, other):
        raise ak._errors.wrap_error(
            ValueError("Input operand 0 does not have enough dimensions ")
        )

    def __truediv__(self, other):
        return self._binop(self._value.__truediv__, other)

    def __floordiv__(self, other):
        return self._binop(self._value.__floordiv__, other)

    def __mod__(self, other):
        return self._binop(self._value.__mod__, other)

    def __divmod__(self, other):
        return self._binop(self._value.__divmod__, other)

    def __pow__(self, other):
        return self._binop(self._value.__pow__, other)

    def __lshift__(self, other):
        return self._binop(self._value.__lshift__, other)

    def __rshift__(self, other):
        return self._binop(self._value.__rshift__, other)

    def __and__(self, other):
        return self._binop(self._value.__and__, other)

    def __xor__(self, other):
        return self._binop(self._value.__xor__, other)

    def __or__(self, other):
        return self._binop(self._value.__or__, other)

    def _rbinop(self, op, other):
        if isinstance(other, JaxScalar):
            return JaxScalar(op(other._value))
        elif isinstance(other, numbers.Number):
            return JaxScalar(op(other))
        elif hasattr(other, "shape") and other.shape == ():
            return JaxScalar(
                getattr(other, op.__name__.replace("__r", "__", 1))(self._value)
            )
        else:
            return NotImplemented

    def __radd__(self, other):
        return self._binop(self._value.__radd__, other)

    def __rsub__(self, other):
        return self._binop(self._value.__rsub__, other)

    def __rmul__(self, other):
        return self._binop(self._value.__rmul__, other)

    def __rmatmul__(self, other):
        raise ak._errors.wrap_error(
            ValueError("Input operand 0 does not have enough dimensions ")
        )

    def __rtruediv__(self, other):
        return self._binop(self._value.__rtruediv__, other)

    def __rfloordiv__(self, other):
        return self._binop(self._value.__rfloordiv__, other)

    def __rmod__(self, other):
        return self._binop(self._value.__rmod__, other)

    def __rdivmod__(self, other):
        return self._binop(self._value.__rdivmod__, other)

    def __rpow__(self, other):
        return self._binop(self._value.__rpow__, other)

    def __rlshift__(self, other):
        return self._binop(self._value.__rlshift__, other)

    def __rrshift__(self, other):
        return self._binop(self._value.__rrshift__, other)

    def __rand__(self, other):
        return self._binop(self._value.__rand__, other)

    def __rxor__(self, other):
        return self._binop(self._value.__rxor__, other)

    def __ror__(self, other):
        return self._binop(self._value.__ror__, other)

    def __iadd__(self, other):
        return self._binop(self._value.__iadd__, other)

    def __isub__(self, other):
        return self._binop(self._value.__isub__, other)

    def __imul__(self, other):
        return self._binop(self._value.__imul__, other)

    def __imatmul__(self, other):
        raise ak._errors.wrap_error(
            ValueError("Input operand 0 does not have enough dimensions ")
        )

    def __itruediv__(self, other):
        return self._binop(self._value.__itruediv__, other)

    def __ifloordiv__(self, other):
        return self._binop(self._value.__ifloordiv__, other)

    def __imod__(self, other):
        return self._binop(self._value.__imod__, other)

    def __idivmod__(self, other):
        return self._binop(self._value.__idivmod__, other)

    def __ipow__(self, other):
        return self._binop(self._value.__ipow__, other)

    def __ilshift__(self, other):
        return self._binop(self._value.__ilshift__, other)

    def __irshift__(self, other):
        return self._binop(self._value.__irshift__, other)

    def __iand__(self, other):
        return self._binop(self._value.__iand__, other)

    def __ixor__(self, other):
        return self._binop(self._value.__ixor__, other)

    def __ior__(self, other):
        return self._binop(self._value.__ior__, other)

    def __neg__(self):
        return JaxScalar(-self._value)

    def __pos__(self):
        return JaxScalar(+self._value)

    def __abs__(self):
        return JaxScalar(abs(self._value))

    def __invert__(self):
        return JaxScalar(~self._value)

    def __complex__(self):
        return complex(self._value)

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __round__(self, n=None):
        return round(self._value, n)

    def __trunc__(self):
        return self._value.__trunc__()

    def __floor__(self):
        return self._value.__floor__()

    def __ceil__(self):
        return self._value.__ceil__()

    def __index__(self):
        return self._value.__index__()

    def _jax_flatten(self):
        return ([self._value], None)

    @classmethod
    def jax_flatten(cls, array):
        assert type(array) is cls
        return array._jax_flatten()

    @classmethod
    def jax_unflatten(cls, aux_data, children):
        assert len(children) == 1
        return JaxScalar(children[0])





# view doesn't support subclasses
# https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.view
class CupyScalar:
    def __init__(self, scalar):
        self._value = scalar
    @property
    def value(self):
        return self._value

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return repr(self._value)

    def __format__(self, format_spec):
        return format(self._value, format_spec)

    def __bytes__(self):
        return bytes(self._value)

    def _compops(self, op, other):
        if isinstance(other, CupyScalar):
            return op(other._value)
        elif isinstance(other, numbers.Number):
            return op(other)
        elif hasattr(other, "shape") and other.shape == ():
            return op(other)
        else:
            return NotImplemented

    def __lt__(self, other):
        return self._compops(self._value.__lt__, other)

    def __le__(self, other):
        return self._compops(self._value.__le__, other)

    def __gt__(self, other):
        return self._compops(self._value.__gt__, other)

    def __ge__(self, other):
        return self._compops(self._value.__ge__, other)

    def __eq__(self, other):
        return self._compops(self._value.__eq__, other)

    def __ne__(self, other):
        return self._compops(self._value.__ne__, other)

    def __hash__(self):
        return hash(self._value)

    def __bool__(self):
        return bool(self._value)

    __slots__ = ["_value"]

    def __len__(self):
        raise ak._errors.wrap_error(TypeError("len() of unsized object"))

    def _binop(self, op, other):
        if isinstance(other, CupyScalar):
            return CupyScalar(op(other._value))
        elif isinstance(other, numbers.Number):
            return CupyScalar(op(other))
        elif hasattr(other, "shape") and other.shape == ():
            return CupyScalar(
                getattr(other, op.__name__.replace("__", "__r", 1))(self._value)
            )
        else:
            return NotImplemented

    def __array__(self):
        import numpy

        return numpy.array(self._value)

    def __add__(self, other):
        return self._binop(self._value.__add__, other)

    def __sub__(self, other):
        return self._binop(self._value.__sub__, other)

    def __mul__(self, other):
        return self._binop(self._value.__mul__, other)

    def __matmul__(self, other):
        raise ak._errors.wrap_error(
            ValueError("Input operand 0 does not have enough dimensions ")
        )

    def __truediv__(self, other):
        return self._binop(self._value.__truediv__, other)

    def __floordiv__(self, other):
        return self._binop(self._value.__floordiv__, other)

    def __mod__(self, other):
        return self._binop(self._value.__mod__, other)

    def __divmod__(self, other):
        return self._binop(self._value.__divmod__, other)

    def __pow__(self, other):
        return self._binop(self._value.__pow__, other)

    def __lshift__(self, other):
        return self._binop(self._value.__lshift__, other)

    def __rshift__(self, other):
        return self._binop(self._value.__rshift__, other)

    def __and__(self, other):
        return self._binop(self._value.__and__, other)

    def __xor__(self, other):
        return self._binop(self._value.__xor__, other)

    def __or__(self, other):
        return self._binop(self._value.__or__, other)

    def _rbinop(self, op, other):
        if isinstance(other, CupyScalar):
            return CupyScalar(op(other._value))
        elif isinstance(other, numbers.Number):
            return CupyScalar(op(other))
        elif hasattr(other, "shape") and other.shape == ():
            return CupyScalar(
                getattr(other, op.__name__.replace("__r", "__", 1))(self._value)
            )
        else:
            return NotImplemented

    def __radd__(self, other):
        return self._binop(self._value.__radd__, other)

    def __rsub__(self, other):
        return self._binop(self._value.__rsub__, other)

    def __rmul__(self, other):
        return self._binop(self._value.__rmul__, other)

    def __rmatmul__(self, other):
        raise ak._errors.wrap_error(
            ValueError("Input operand 0 does not have enough dimensions ")
        )

    def __rtruediv__(self, other):
        return self._binop(self._value.__rtruediv__, other)

    def __rfloordiv__(self, other):
        return self._binop(self._value.__rfloordiv__, other)

    def __rmod__(self, other):
        return self._binop(self._value.__rmod__, other)

    def __rdivmod__(self, other):
        return self._binop(self._value.__rdivmod__, other)

    def __rpow__(self, other):
        return self._binop(self._value.__rpow__, other)

    def __rlshift__(self, other):
        return self._binop(self._value.__rlshift__, other)

    def __rrshift__(self, other):
        return self._binop(self._value.__rrshift__, other)

    def __rand__(self, other):
        return self._binop(self._value.__rand__, other)

    def __rxor__(self, other):
        return self._binop(self._value.__rxor__, other)

    def __ror__(self, other):
        return self._binop(self._value.__ror__, other)

    def __iadd__(self, other):
        return self._binop(self._value.__iadd__, other)

    def __isub__(self, other):
        return self._binop(self._value.__isub__, other)

    def __imul__(self, other):
        return self._binop(self._value.__imul__, other)

    def __imatmul__(self, other):
        raise ak._errors.wrap_error(
            ValueError("Input operand 0 does not have enough dimensions ")
        )

    def __itruediv__(self, other):
        return self._binop(self._value.__itruediv__, other)

    def __ifloordiv__(self, other):
        return self._binop(self._value.__ifloordiv__, other)

    def __imod__(self, other):
        return self._binop(self._value.__imod__, other)

    def __idivmod__(self, other):
        return self._binop(self._value.__idivmod__, other)

    def __ipow__(self, other):
        return self._binop(self._value.__ipow__, other)

    def __ilshift__(self, other):
        return self._binop(self._value.__ilshift__, other)

    def __irshift__(self, other):
        return self._binop(self._value.__irshift__, other)

    def __iand__(self, other):
        return self._binop(self._value.__iand__, other)

    def __ixor__(self, other):
        return self._binop(self._value.__ixor__, other)

    def __ior__(self, other):
        return self._binop(self._value.__ior__, other)

    def __neg__(self):
        return CupyScalar(-self._value)

    def __pos__(self):
        return CupyScalar(+self._value)

    def __abs__(self):
        return CupyScalar(abs(self._value))

    def __invert__(self):
        return CupyScalar(~self._value)

    def __complex__(self):
        return complex(self._value)

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __round__(self, n=None):
        return round(self._value, n)

    def __trunc__(self):
        return self._value.__trunc__()

    def __floor__(self):
        return self._value.__floor__()

    def __ceil__(self):
        return self._value.__ceil__()

    def __index__(self):
        return self._value.__index__()


# Numpy Scalars are perfectly compatible with all Numpy Functions
import numpy as np


class NumpyScalar(np.ndarray):
    def __new__(cls, value):
        return np.array(value).view(NumpyScalar)

    def __init__(self, scalar):
        self._value = np.array(scalar)

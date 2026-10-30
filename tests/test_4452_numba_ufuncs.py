# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")
ak.numba.register_and_check()


@pytest.mark.parametrize(
    "function",
    [
        np.sin,
        np.cos,
        np.exp,
        np.sqrt,
        np.negative,
        np.absolute,
        np.isnan,
        np.isfinite,
        np.logical_not,
    ],
)
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_unary(function, dtype):
    @numba.njit
    def evaluate(a):
        return function(a)

    values = np.array([1, 2, 3], dtype=dtype)
    np.testing.assert_allclose(evaluate(ak.Array(values)), evaluate(values), rtol=1e-6)


@pytest.mark.parametrize(
    "function",
    [
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
        np.maximum,
        np.equal,
        np.less,
        np.logical_and,
    ],
)
@pytest.mark.parametrize("kind", ["both", "left", "right", "numpy_left", "numpy_right"])
def test_binary(function, kind):
    @numba.njit
    def evaluate(a, b):
        return function(a, b)

    left = np.array([1.0, 2.0, 3.0])
    right = np.array([3.0, 2.0, 1.0])
    a, b = ak.Array(left), ak.Array(right)
    if kind == "left":
        b, right = 2.0, 2.0
    elif kind == "right":
        a, left = 2.0, 2.0
    elif kind == "numpy_left":
        a = left
    elif kind == "numpy_right":
        b = right
    np.testing.assert_allclose(evaluate(a, b), evaluate(left, right))


def test_fused_expression():
    @numba.njit
    def evaluate(a, b):
        return np.subtract(np.sin(a) * 2 + np.cos(a) + np.sqrt(b), b)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 9.0, 16.0])
    np.testing.assert_allclose(evaluate(ak.Array(a), ak.Array(b)), evaluate(a, b))


@pytest.mark.parametrize("keyword", [False, True])
def test_out(keyword):
    @numba.njit
    def evaluate(a, out):
        if keyword:
            return np.sin(a, out=out)
        return np.sin(a, out)

    a = ak.Array([1.0, 2.0, 3.0])
    out = np.zeros(3)
    result = evaluate(a, out)
    np.testing.assert_allclose(result, np.sin(np.array([1.0, 2.0, 3.0])))
    assert np.shares_memory(result, out)


def test_subarray_and_slice():
    @numba.njit
    def evaluate(a):
        return np.sin(a[1][1:])

    np.testing.assert_allclose(
        evaluate(ak.Array([[0.0], [99.0, 1.0, 2.0]])), np.sin([1.0, 2.0])
    )


@pytest.mark.parametrize("data", [[[1], [2]], [1, None], ["a", "b"], [{"x": 1}]])
def test_unsupported_layouts(data):
    @numba.njit
    def evaluate(a):
        return np.sin(a)

    with pytest.raises(numba.core.errors.TypingError):
        evaluate(ak.Array(data))


def test_no_rewrites():
    @numba.njit(no_rewrites=True)
    def evaluate(a):
        return np.sin(a)

    np.testing.assert_allclose(evaluate(ak.Array([1.0, 2.0])), np.sin([1.0, 2.0]))


def supported_ufunc_cases():
    from numba.np import ufunc_db

    for function in sorted(ufunc_db.get_ufuncs(), key=lambda f: f.__name__):
        if function.nin not in (1, 2) or function.nout != 1:
            continue
        for signature in ufunc_db.get_ufunc_info(function):
            inputs = signature.split("->")[0]
            if all(char in "?bBhHiIlLqQfdFD" for char in inputs):
                yield pytest.param(function, inputs, id=function.__name__)
                break


@pytest.mark.parametrize(("function", "dtypes"), list(supported_ufunc_cases()))
def test_native_ufunc_inventory(function, dtypes):
    # Exercise every unary/binary single-output ufunc in Numba's native database,
    # using an input loop supported by the installed Numba version.
    values = [np.array([1, 2, 3], dtype=char) for char in dtypes]
    if len(values) == 1:

        @numba.njit
        def evaluate(a):
            return function(a)

    else:

        @numba.njit
        def evaluate(a, b):
            return function(a, b)

    expected = evaluate(*values)
    actual = evaluate(*(ak.Array(value) for value in values))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex(dtype):
    @numba.njit
    def evaluate(a):
        return np.sin(a) + np.conjugate(a)

    values = np.array([1 + 2j, 3 - 4j], dtype=dtype)
    np.testing.assert_allclose(evaluate(ak.Array(values)), evaluate(values), rtol=1e-6)


def test_broadcast_numpy_operand():
    @numba.njit
    def evaluate(a, b):
        return np.add(a, b)

    values = np.array([1.0, 2.0, 3.0])
    matrix = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(evaluate(ak.Array(values), matrix), values + matrix)
    with pytest.raises(ValueError):
        evaluate(ak.Array(values), np.zeros((2, 4)))


def test_empty():
    @numba.njit
    def evaluate(a):
        return np.sin(a)

    assert evaluate(ak.Array(np.array([], dtype=np.float64))).shape == (0,)


def test_awkward_output_is_not_implicitly_converted():
    @numba.njit
    def evaluate(a, out):
        return np.add(a, 2.0, out)

    with pytest.raises(numba.core.errors.TypingError):
        evaluate(ak.Array([1.0, 2.0, 3.0]), ak.Array(np.zeros(3)))


def test_parallel():
    @numba.njit(parallel=True)
    def evaluate(a):
        return np.sin(a) + np.cos(a)

    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        evaluate(ak.Array(values)), np.sin(values) + np.cos(values)
    )


def test_datetime():
    @numba.njit
    def evaluate(a):
        return np.isnat(a)

    values = np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(evaluate(ak.Array(values)), np.isnat(values))


def test_control_flow():
    @numba.njit
    def evaluate(a, use_sine):
        if use_sine:
            return np.sin(a) + np.cos(a)
        return np.sqrt(a) * 2

    values = np.array([1.0, 2.0, 3.0])
    for use_sine in (True, False):
        np.testing.assert_allclose(
            evaluate(ak.Array(values), use_sine), evaluate(values, use_sine)
        )


def test_python_overload_implementation_bodies(monkeypatch):
    from awkward._connect.numba import ufunc as overloads

    factories = []

    def capture(function):
        def register(factory):
            factories.append(factory)
            return factory

        return register

    monkeypatch.setattr(numba.extending, "overload", capture)
    overloads._register_unary(np.sin)
    overloads._register_binary(np.add)
    array = ak.Array([1.0, 2.0])
    viewtype = array.numba_type
    output = np.empty(2)
    outtype = numba.typeof(output)
    for out in (None, numba.types.none):
        unary = factories[0](viewtype, out)
        binary = factories[1](viewtype, numba.types.float64, out)
        np.testing.assert_allclose(np.asarray(unary(array)), np.sin([1, 2]))
        np.testing.assert_allclose(np.asarray(binary(array, 2)), [3, 4])
    unary = factories[0](viewtype, outtype)
    binary = factories[1](viewtype, numba.types.float64, outtype)
    assert unary(np.asarray(array), output) is output
    assert binary(np.asarray(array), 2, output) is output
    np.testing.assert_array_equal(output, [3, 4])
    convert = overloads._overload_as_numpy(viewtype)
    np.testing.assert_array_equal(convert(array), [1, 2])
    unchanged = overloads._overload_as_numpy(outtype)
    assert unchanged(output) is output
    assert overloads._as_numpy(output) is output
    assert overloads._replace_variables(42, {}) == 42

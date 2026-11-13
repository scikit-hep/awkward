# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")
ak.numba.register_and_check()

# The fixture temporarily changes Numba's process-wide compiler configuration.
pytestmark = pytest.mark.thread_unsafe


@pytest.fixture(autouse=True)
def report_compiled_coverage(monkeypatch):
    # Python's tracer cannot see overload bodies executed as native code. Let
    # Numba report lowered source lines when coverage is active. The tests still
    # execute and check the compiled results; this is not runtime branch coverage.
    from numba.core import config

    if hasattr(config, "JIT_COVERAGE"):
        monkeypatch.setattr(config, "JIT_COVERAGE", True)


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


@pytest.mark.parametrize("function", [np.sin, np.add])
@pytest.mark.parametrize("keyword", [False, True])
def test_explicit_none_output_is_rejected(function, keyword):
    if function.nin == 1:

        @numba.njit
        def evaluate(a):
            if keyword:
                return function(a, out=None)
            return function(a, None)

    else:

        @numba.njit
        def evaluate(a):
            if keyword:
                return function(a, 2.0, out=None)
            return function(a, 2.0, None)

    values = np.array([1.0, 2.0])
    for value in (values, ak.Array(values)):
        with pytest.raises((numba.core.errors.TypingError, NotImplementedError)):
            evaluate(value)


@pytest.mark.parametrize("explicit_outputs", [False, True])
@pytest.mark.parametrize("no_rewrites", [False, True])
@pytest.mark.parametrize("kind", ["left", "right", "both", "numpy_left", "numpy_right"])
def test_divmod_outputs(explicit_outputs, no_rewrites, kind):
    @numba.njit(no_rewrites=no_rewrites)
    def evaluate(a, b, quotient, remainder):
        if explicit_outputs:
            return np.divmod(a, b, quotient, remainder)
        return np.divmod(a, b)

    values = np.array([-3.5, 1.5, 8.0])
    left, right = values, np.array([2.0, 3.0, 4.0])
    a, b = ak.Array(left), ak.Array(right)
    if kind == "left":
        b, right = 2.0, 2.0
    elif kind == "right":
        a, left = 2.0, 2.0
    elif kind == "numpy_left":
        a = left
    elif kind == "numpy_right":
        b = right
    quotient, remainder = np.empty(3), np.empty(3)
    expected = evaluate(left, right, np.empty(3), np.empty(3))
    actual = evaluate(a, b, quotient, remainder)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    if explicit_outputs:
        assert np.shares_memory(actual[0], quotient)
        assert np.shares_memory(actual[1], remainder)


@pytest.mark.parametrize("function", [np.modf, np.frexp, np.matmul])
def test_unsupported_ufunc_does_not_gain_awkward_overload(function):
    from numba.np import ufunc_db

    if function in ufunc_db.get_ufuncs():
        pytest.skip("Numba now implements this ufunc")
    if function.nin == 1:

        @numba.njit
        def evaluate(a):
            return function(a)

    else:

        @numba.njit
        def evaluate(a):
            return function(a, a)

    with pytest.raises(numba.core.errors.TypingError) as error:
        evaluate(np.array([1.0, 2.0]))
    assert "awkward/_connect/numba/ufunc.py" not in str(error.value).replace("\\", "/")


def test_parallel_control_flow_and_multiple_expressions():
    @numba.njit(parallel=True)
    def evaluate(a):
        sine = np.sin(a)
        cosine = np.cos(a)
        result = sine + cosine
        for i in numba.prange(len(result)):
            if i % 2:
                result[i] *= 2
        return result + np.sqrt(a)

    values = np.arange(1.0, 33.0)
    np.testing.assert_allclose(evaluate(ak.Array(values)), evaluate(values))


def test_rewrite_preserves_input_ir_and_definitions(monkeypatch):
    from numba.core import ir

    from awkward._connect.numba.ufunc import ConvertUfuncArrayViews

    apply = ConvertUfuncArrayViews.apply
    checked = []

    def inspect_rewrite(self):
        originals = [
            (instruction, instruction.value, instruction.value.expr)
            for instruction in self.block.find_insts(ir.Assign)
            if isinstance(instruction.value, ir.Expr)
            and instruction.value.op == "arrayexpr"
        ]
        rewritten = apply(self)
        for instruction, value, expression in originals:
            assert instruction.value is value
            assert instruction.value.expr is expression
        for instruction in rewritten.find_insts(ir.Assign):
            if instruction.target.name.startswith(("ak_asarray", "ak_numpy_view")):
                assert (
                    self.func_ir.get_definition(instruction.target) is instruction.value
                )
                checked.append(instruction.target.name)
            elif instruction.target.name in self.matches:
                assert (
                    self.func_ir.get_definition(instruction.target) is instruction.value
                )
        return rewritten

    monkeypatch.setattr(ConvertUfuncArrayViews, "apply", inspect_rewrite)

    @numba.njit
    def evaluate(a):
        return np.sin(a) + np.cos(a)

    values = np.arange(1.0, 5.0)
    np.testing.assert_allclose(
        evaluate(ak.Array(values)), np.sin(values) + np.cos(values)
    )
    assert checked


@pytest.mark.parametrize("awkward_output", [False, True])
def test_divmod_invalid_outputs(awkward_output):
    @numba.njit
    def evaluate(a, output):
        if awkward_output:
            return np.divmod(a, 2.0, output, np.empty(2))
        return np.divmod(a, 2.0, output)

    values = ak.Array([1.0, 2.0])
    output = ak.Array(np.empty(2)) if awkward_output else np.empty(2)
    with pytest.raises(numba.core.errors.TypingError):
        evaluate(values, output)


def test_numpy_only_expression_does_not_apply_conversion(monkeypatch):
    from awkward._connect.numba.ufunc import ConvertUfuncArrayViews

    def unexpected_conversion(self):
        pytest.fail("NumPy-only IR must not use the Awkward conversion rewrite")

    monkeypatch.setattr(ConvertUfuncArrayViews, "apply", unexpected_conversion)

    @numba.njit
    def evaluate(a):
        return np.sin(a) + np.cos(a)

    values = np.arange(1.0, 5.0)
    np.testing.assert_allclose(evaluate(values), np.sin(values) + np.cos(values))


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("no_rewrites", [False, True])
def test_binary_numpy_output(keyword, no_rewrites):
    @numba.njit(no_rewrites=no_rewrites)
    def evaluate(a, b, output):
        if keyword:
            return np.add(a, b, out=output)
        return np.add(a, b, output)

    values = np.array([1.0, 2.0, 3.0])
    output = np.full(5, -99.0)
    result = evaluate(ak.Array(values), 2.0, output[1:4])
    np.testing.assert_array_equal(result, values + 2)
    np.testing.assert_array_equal(output, [-99, 3, 4, 5, -99])
    assert np.shares_memory(result, output)


@pytest.mark.parametrize("data", [[[1], [2]], [1, None], ["a", "b"]])
def test_divmod_unsupported_layouts(data):
    @numba.njit
    def evaluate(a):
        return np.divmod(a, 2.0)

    with pytest.raises(numba.core.errors.TypingError):
        evaluate(ak.Array(data))

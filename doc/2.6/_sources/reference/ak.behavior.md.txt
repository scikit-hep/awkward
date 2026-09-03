# ak.behavior

```{py:data} ak.behavior
```

## Motivation

A data structure is defined both in terms of the information it encodes and
in how it can be used. For example, a hash-table is not just a buffer, it's
also the "get" and "set" operations that make the buffer usable as a key-value
store. Awkward Arrays have a suite of operations for transforming tree
structures into new tree structures, but an application of these structures to
a data analysis problem should be able to interpret them as objects in the
analysis domain, such as latitude-longitude coordinates in geographical
studies or Lorentz vectors in particle physics.

Object-oriented programming unites data with its operations. This is a
conceptual improvement for data analysts because functions like "distance
between this latitude-longitude point and another on a spherical globe" can
be bound to the objects that represent latitude-longitude points. It
matches the way that data analysts usually think about their data.

However, if these methods are saved in the data, or are written in a way
that will only work for one version of the data structures, then it becomes
difficult to work with large datasets. Old data that do not "fit" the new
methods would have to be converted, or the analysis would have to be broken
into different cases for each data generation. This problem is known as
schema evolution, and there are many solutions to it.

The approach taken by the Awkward Array library is to encode very little
interpretation into the data themselves and apply an interpretation as
late as possible. Thus, a latitude-longitude record might be stamped with
the name `"latlon"`, but the operations on it are added immediately before
the user wants them. These operations can be written in such a way that
they only require the `"latlon"` to have `lat` and `lon` fields, so
different versions of the data can have additional fields or even be
embedded in different structures.

## Parameters and behaviors

In Awkward Array, metadata are embedded in data using an array node's
**parameters**, and parameter-dependent operations can be defined using
**behavior**. A global mapping from parameters to behavior is in a dict called
{data}`.behavior`:

```python
>>> import awkward as ak
>>> ak.behavior
```

but behavior dicts can also be loaded into {class}`ak.Array`,
{class}`ak.Record`, and {class}`ak.ArrayBuilder` objects as a
constructor argument. See
{attr}`ak.Array.behavior`.

The general flow is

- **parameters** link data objects to names;
- **behavior** links names to code.

In large datasets, parameters may be hard to change (permanently, at least:
on-the-fly parameter changes are easier), but behavior is easy to change
(it is always assigned on-the-fly).

In the following example, we create two nested arrays of records with fields
`"x"` and `"y"` and the records are named `"point"`.

```python
one = ak.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                [],
                [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
                [{"x": 6, "y": 6.6}],
                [{"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}]],
               with_name="point")
two = ak.Array([[{"x": 0.9, "y": 1}, {"x": 2, "y": 2.2}, {"x": 2.9, "y": 3}],
                [],
                [{"x": 3.9, "y": 4}, {"x": 5, "y": 5.5}],
                [{"x": 5.9, "y": 6}],
                [{"x": 6.9, "y": 7}, {"x": 8, "y": 8.8}, {"x": 8.9, "y": 9}]],
               with_name="point")
```

The name appears in the way the type is presented as a string (a departure from
[Datashape notation](https://datashape.readthedocs.io/)):

```python
>>> ak.type(one)
5 * var * point["x": int64, "y": float64]
```

and it may be accessed as the `"__record__"` property, through the
{attr}`ak.Array.layout`:

```python
>>> one.layout
<ListOffsetArray64>
    <offsets><Index64 i="[0 3 3 5 6 9]" offset="0" length="6"/></offsets>
    <content><RecordArray>
        <parameters>
            <param key="__record__">"point"</param>
        </parameters>
        <field index="0" key="x">
            <NumpyArray format="l" shape="9" data="1 2 3 4 5 6 7 8 9"/>
        </field>
        <field index="1" key="y">
            <NumpyArray format="d" shape="9" data="1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9"/>
        </field>
    </RecordArray></content>
</ListOffsetArray64>
>>> one.layout.content.parameters
{'__record__': 'point'}
```

We have to dig into the layout's content because the `"__record__"` parameter
is set on the {class}`ak.contents.RecordArray`, which is buried inside of a
{class}`ak.contents.ListOffsetArray`.

Alternatively, we can navigate to a single {class}`ak.Record` first:

```python
>>> one[0, 0]
<Record {x: 1, y: 1.1} type='point["x": int64, "y": float64]'>
>>> one[0, 0].layout.parameters
{'__record__': 'point'}
```

## Adding behavior to records

Suppose we want the points in the above example to be able to calculate
distances to other points. We can do this by creating a subclass of
{class}`ak.Record` that has the new methods and associating it with
the `"__record__"` name.

```python
class Point(ak.Record):
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

ak.behavior["point"] = Point
```

Now `one[0, 0]` is instantiated as a `Point`, rather than a {class}`ak.Record`,

```python
>>> one[0, 0]
<Point {x: 1, y: 1.1} type='point["x": int64, "y": float64]'>
```

and it has the `distance` method.

```python
>>> for xs, ys in zip(one, two):
...     for x, y in zip(xs, ys):
...         print(x.distance(y))
0.14142135623730953
0.0
0.31622776601683783
0.4123105625617664
0.0
0.6082762530298216
0.7071067811865477
0.0
0.905538513813742
```

Looping over data in Python is inconvenient and slow; we want to compute
quantities like this with array-at-a-time methods, but `distance` is
bound to a {class}`ak.Record`, not an {class}`ak.Array` of records.

```python
>>> one.distance(two)
AttributeError: no field named 'distance'
```

To add `distance` as a method on arrays of points, create a subclass of
{class}`ak.Array` and attach that as `ak.behavior["*", "point"]` for
"array of points."

```python
class PointArray(ak.Array):
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

ak.behavior["*", "point"] = PointArray
```

Now `one[0]` is a `PointArray` and can compute `distance` on arrays at a
time. Thanks to NumPy's
[universal function](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
(ufunc) syntax, the expression is the same (and could perhaps be implemented
once and used by both `Point` and `PointArray`).

```python
>>> one[0]
<PointArray [{x: 1, y: 1.1}, ... {x: 3, y: 3.3}] type='3 * point["x": int64, "y"...'>
>>> one[0].distance(two[0])
<Array [0.141, 0, 0.316] type='3 * float64'>
```

`one` itself is also a `PointArray`, and the same applies.

```python
>>> one
<Array [[{x: 1, y: 1.1}, ... x: 9, y: 9.9}]] type='5 * var * point["x": int64, "...'>
>>> one.distance(two)
[[0.141, 0, 0.316],
 [],
 [0.412, 0],
 [0.608],
 [0.707, 0, 0.906]]
```

One last caveat: our `one` array was created *before* this behavior was
assigned, so it needs to be recreated to be a member of the new class. The
normal {class}`ak.Array` constructor is sufficient for this. This is only
an issue if you're working interactively (but something to think about when
debugging!).

```python
>>> one = ak.Array(one)
>>> two = ak.Array(two)
```

Now it works, and again we're taking advantage of the fact that the expression
for `distance` based on ufuncs works equally well on Awkward Arrays.

```python
>>> one
<PointArray [[{x: 1, y: 1.1}, ... x: 9, y: 9.9}]] type='5 * var * point["x": int...'>
>>> one.distance(two)
<Array [[0.141, 0, 0.316, ... 0.707, 0, 0.906]] type='5 * var * float64'>
```

**In most cases, you want to apply array-of-records for all levels of list-depth:** use `ak.behavior["*", record_name]`.

## Overriding NumPy ufuncs and binary operators

The {class}`ak.Array` class overrides Python's binary operators with the
equivalent ufuncs, so `__eq__` actually calls {data}`numpy.equal`, for instance.
This is also true of other basic functions, like `__abs__` for overriding
{func}`abs` with {data}`numpy.absolute`. Each ufunc is then passed down to the leaves
(deepest sub-elements) of an Awkward data structure.

For example,

```python
>>> ak.Array([[1, 2, 3], [], [4]]) == ak.Array([[3, 2, 1], [], [4]])
<Array [[False, True, False], [], [True]] type='3 * var * bool'>
```

However, this does not apply to records or named types until they are explicitly
overridden:

```python
>>> one == two
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
...
ValueError: no overloads for custom types: equal(point, point)
```

We might want to take an object-oriented view in which the `==` operation
applies to points, regardless of how deeply they are nested. If we try to do
it by adding `__eq__` as a method on `PointArray`, it would work if the
`PointArray` is the top of the data structure, but not if it's nested within
another structure.

Instead, we should override {data}`numpy.equal` itself. Custom ufunc overrides are
checked at every step in broadcasting, so the override would be applied if
point objects are discovered at any level.

```python
def point_equal(left, right):
    return np.logical_and(left.x == right.x, left.y == right.y)

ak.behavior[np.equal, "point", "point"] = point_equal
```

The above should be read as "override :data\`np.equal\` for cases in which both
arguments are `"point"`."

```python
>>> ak.to_list(one == two)
[[False, True, False], [], [False, True], [False], [False, True, False]]
```

Similarly for overriding {func}`abs`

```python
>>> def point_abs(point):
...     return np.sqrt(point.x**2 + point.y**2)
...
>>> ak.behavior[np.absolute, "point"] = point_abs
>>> ak.to_list(abs(one))
[[1.4866068747318506, 2.973213749463701, 4.459820624195552],
 [],
 [5.946427498927402, 7.433034373659253],
 [8.919641248391104],
 [10.406248123122953, 11.892854997854805, 13.379461872586655]]
```

and all other ufuncs.

If you need a placeholder for "any number," use {class}`numbers.Real`,
{class}`numbers.Integral`, etc. Non-arrays are resolved by type; builtin Python
numbers and NumPy numbers are subclasses of the generic number types in the
{mod}`numbers` library.

Also, for commutative operations, be sure to override both operator orders.
(Function signatures are matched to {data}`ak.behavior` using multiple dispatch.)

```python
>>> import numbers
>>> def point_lmult(point, scalar):
...     return ak.Array({"x": point.x * scalar, "y": point.y * scalar})
...
>>> def point_rmult(scalar, point):
...     return point_lmult(point, scalar)
...
>>> ak.behavior[np.multiply, "point", numbers.Real] = point_lmult
>>> ak.behavior[np.multiply, numbers.Real, "point"] = point_rmult
>>> ak.to_list(one * 10)
[[{'x': 10, 'y': 11.0}, {'x': 20, 'y': 22.0}, {'x': 30, 'y': 33.0}],
 [],
 [{'x': 40, 'y': 44.0}, {'x': 50, 'y': 55.0}],
 [{'x': 60, 'y': 66.0}],
 [{'x': 70, 'y': 77.0}, {'x': 80, 'y': 88.0}, {'x': 90, 'y': 99.0}]]
```

If you need to override ufuncs in more generality, you can use the
{class}`numpy.ufunc` interface:

```python
>>> def apply_ufunc(ufunc, method, args, kwargs):
...     if ufunc in (np.sin, np.cos, np.tan):
...         x = ufunc(args[0].x)
...         y = ufunc(args[0].y)
...         return ak.Array({"x": x, "y": y})
...     else:
...         return NotImplemented
...
>>> ak.behavior[np.ufunc, "point"] = apply_ufunc
>>> ak.to_list(np.sin(one))
[[{'x': 0.8414709848078965, 'y': 0.8912073600614354},
  {'x': 0.9092974268256817, 'y': 0.8084964038195901},
  {'x': 0.1411200080598672, 'y': -0.1577456941432482}],
 [],
 [{'x': -0.7568024953079282, 'y': -0.951602073889516},
  {'x': -0.9589242746631385, 'y': -0.7055403255703919}],
 [{'x': -0.27941549819892586, 'y': 0.31154136351337786}],
 [{'x': 0.6569865987187891, 'y': 0.9881682338770004},
  {'x': 0.9893582466233818, 'y': 0.5849171928917617},
  {'x': 0.4121184852417566, 'y': -0.45753589377532133}]]
>>> np.sqrt(one)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
...
ValueError: no overloads for custom types: sqrt(point)
```

But be forewarned: the `ak.behavior[np.ufunc, name]` syntax will match
*any* ufunc that has an array containing an array with type `name`
*anywhere* in the argument list. The first array in the argument list
with type `name` will be matched instead of more detailed argument lists
with type `name` at a later spot in the list. The "apply_ufunc" interface
is *greedy*.

## Overriding NumPy reducers

In addition to ufuncs, it is also possible to override _reducers_ on records. Consider a 2D vector that implements binary addition:

```python
def vector_add(left, right):
    return ak.contents.RecordArray(
        [
            ak.to_layout(left["rho"] + right["rho"]),
            ak.to_layout(left["phi"] + right["phi"]),
        ],
        ["rho", "phi"],
        parameters={"__record__": "Vector2D"},
    )


ak.behavior[np.add, "Vector2D", "Vector2D"] = vector_add
```

Whilst the `np.add` overload permits binary addition of `Vector2D` objects,

```pycon 
>>> vector = ak.Array(
...     [[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}], [{"rho": -2.2, "phi": 0.0}, {"rho": 3.1, "phi": 0.9}]],
...     with_name="Vector2D",
... )
>>> (vector + vector).show()
[[{rho: -2.2, phi: -0.2}, {rho: 2.2, phi: 0.2}],
 [{rho: -4.4, phi: 0}, {rho: 6.2, phi: 1.8}]]
```

it does not permit the use of the `ak.sum` reducer:

```pycon
>>> ak.sum(vector, axis=-1)
TypeError: no ak.sum overloads for custom types: rho, phi

This error occurred while calling

    ak.sum(
        array = <Array [[{rho: -1.1, ...}, ...], ...] type='2 * var * Vecto...'>
        axis = -1
        keepdims = False
        mask_identity = False
        highlevel = True
        behavior = None
    )
```

To implement support for reducers like `ak.sum`, we should override them with a behavior:

```pycon 
>>> def vector_sum(vector, mask_identity):
...     return ak.contents.RecordArray(
...         [
...             ak.sum(vector["rho"], highlevel=False, axis=-1),
...             ak.sum(vector["phi"], highlevel=False, axis=-1),
...         ],
...         ["rho", "phi"],
...         parameters={"__record__": "Vector2D"},
...     )
>>> ak.behavior[ak.sum, "Vector2D"] = vector_sum
>>> ak.sum(vector, axis=-1).show()
[{rho: 0, phi: 0},
{rho: 0.9, phi: 0.9}]
```

Custom reducers are invoked with two arguments: a 2D array containing lists of records, and a boolean `mask_identity` indicating whether the reducer should introduce an option type (for reductions along empty sublists). If the reducer does not introduce an option type, and `mask=True`, Awkward will mask the result at the appropriate positions. The reducer should return an {class}`ak.Array` or {class}`ak.contents.Content` with the same number of elements as the input array. The reduction itself should be performed along `axis=1`, dropping the reduced dimension (i.e. `keepdims=False`). 


## Mixin decorators

The pattern of adding additional properties and function overrides to records
and arrays of records is quite common, and can be nicely described by the "mixin"
idiom: a class with no constructor that is mixed with both the {class}`ak.Array` and {class}`ak.Record`
class as to create new derived classes. The {func}`ak.mixin_class` and {func}`ak.mixin_class_method`
python decorators assist with some of this boilerplate. Consider the `Point` class
from above; we can implement all the functionality so far described as follows:

```python
@ak.mixin_class(ak.behavior)
class Point:
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @ak.mixin_class_method(np.equal, {"Point"})
    def point_equal(self, other):
        return np.logical_and(self.x == other.x, self.y == other.y)

    @ak.mixin_class_method(np.abs)
    def point_abs(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)
```

The behavior name is taken as the mixin class name, e.g. here it is `Point` (as opposed
to lowercase `point` previously). We can extend our implementation to allow `Point` types
to be added by overriding the `np.add` ufunc (appending to our class definition):

```python
class Point:
    # ...

    @ak.mixin_class_method(np.add, {"Point"})
    def point_add(self, other):
        return ak.zip(
            {"x": self.x + other.x, "y": self.y + other.y}, with_name="Point",
        )
```

The real power of using mixin classes comes from the ability to inherit behaviors.
Consider a `Point`-like record that also has a `weight` field. Suppose that we want
these `WeightedPoint` types to have the same distance and magnitude functionality, but
only be considered equal when they have the same weight. Also, suppose we want the addition
of two weighted points to give their weighted mean rather than a sum. We could implement
such a class as follows:

```python
@ak.mixin_class(ak.behavior)
class WeightedPoint(Point):
    @ak.mixin_class_method(np.equal, {"WeightedPoint"})
    def weighted_equal(self, other):
        return np.logical_and(self.point_equal(other), self.weight == other.weight)

    @ak.mixin_class_method(np.add, {"WeightedPoint"})
    def weighted_add(self, other):
        sumw = self.weight + other.weight
        return ak.zip(
            {
                "x": (self.x * self.weight + other.x * other.weight) / sumw,
                "y": (self.y * self.weight + other.y * other.weight) / sumw,
                "weight": sumw,
            },
            with_name="WeightedPoint",
        )
```

A footnote: in this implementation, adding a WeightedPoint and a Point returns a Point.
One may wish to disable this by type-checking, since the functionalities are rather different.

## Adding behavior to arrays

Occasionally, you may want to add behavior to an array that does not contain
records. Historically, this mechanism was used to help implement strings (although)
strings have since been made a part of Awkward's internals, alongside categorical types.

Awkward Array supports adding behaviors to the list-types in an array. Let's suppose that 
one wishes to define a special list that defines a `reversed()` method, equivalent to `array[..., ::-1]`.

First, one must define the behavior class
```python
class ReversibleArray(ak.Array):
    def reversed(self):
        return self[..., ::-1]
```
To make this behavior class available to new arrays, it must be associated with its name
```python
ak.behavior["reversible"] = ReversibleArray
```
One can then add the `__list__` parameter to a list-type array to request this behavior class
```pycon
>>> reversible_list = ak.with_parameter([[1, 2, 3], [4], [5, 6, 7]], "__list__", "reversible")
>>> reversible_list.reversed()
[[3, 2, 1], [4], [7, 6, 5]]
```
If this list is nested within another list, the array class will not be found:
```pycon
>>> nested_list = ak.unflatten(reversible_list, [2, 1])
>>> nested_list.reversed()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
...
AttributeError: no field named 'reversed'
```

Just like with records, we can register a "deep" array class that will be found for all levels of list-depth, using `"*"`:
```python
>>> ak.behavior["*", "reversible"] = ReversibleArray
>>> nested_list = ak.unflatten(reversible_list, [2, 1])
>>> nested_list.reversed()
[[[3, 2, 1], [4]], [[7, 6, 5]]]
```

In `ak.behaviors.string`, string behaviors are assigned with lines like

```python
ak.behavior["string"] = StringBehavior
ak.behavior[np.equal, "string", "string"] = _string_equal
```

## Custom type names

To change the type-string representation of our custom list, a
`"__typestr__"` behavior can be registered:
```python
ak.behavior["__typestr__", "reversible"] = "a-reversible-list"
```

so that

```python
>>> ak.type(ak.with_parameter([[1, 2, 3], [4], [5, 6, 7]], "__list__", "reversible"))
3 * a-reversible-list
```

## Overriding behavior in Numba

Awkward Arrays can be arguments and return values of functions compiled with
[Numba](http://numba.pydata.org). Since these functions run on low-level
objects, most functionality must be reimplemented, including behavioral
overrides.

The documentation on
[Extending Numba](https://numba.pydata.org/numba-doc/dev/extending/index.html)
introduces **typing**, **lowering**, and **models**, which are necessary for
reimplementing the behavior of a Python object in the compiled environment.
To apply the same to records and arrays from an Awkward data structure, we
use {data}`ak.behavior` hooks that start with `"__numba_typer__"` and
`"__numba_lower__"`.

**Case 1:** Adding a property, such as `rec.property_name`.

```python
ak.behavior["__numba_typer__", record_name, property_name] = typer
ak.behavior["__numba_lower__", record_name, property_name] = lower
```

The `typer` function takes an
{func}`ak._connect._numba.arrayview.ArrayViewType` as its only argument
and returns the property's type.

The `lower` function takes the standard `context, builder, sig, args`
arguments and returns the lowered value. Given a Python `function` that
takes one record and returns the property, the `lower` can be

```python
def lower(context, builder, sig, args):
    return context.compile_internal(builder, function, sig, args)
```

**Case 2:** Adding a method, such as `rec.method_name(arg0, arg1)`.

```python
ak.behavior["__numba_typer__", record_name, method_name, ()] = typer
ak.behavior["__numba_lower__", record_name, method_name, ()] = lower
```

The last item is an *empty* tuple, `()` (regardless of whether the method
takes any arguments).

In this case, the `typer` takes an
{func}`ak._connect._numba.arrayview.ArrayViewType` as well as any arguments
and returns the property's type, and the `sig` and `args` in `lower`
include these arguments.

**Case 3:** Unary and binary operations, like `-rec1` and `rec1 + rec2`.

```python
ak.behavior["__numba_typer__", operator.neg, "rec1"] = typer
ak.behavior["__numba_lower__", operator.neg, "rec1"] = lower

ak.behavior["__numba_typer__", "rec1", operator.add, "rec2"] = typer
ak.behavior["__numba_lower__", "rec1", operator.add, "rec2"] = lower
```

**Case 4:** Completely replacing the Awkward record with an object in Numba.

If a fully defined model for the object already exists and Numba, we can
have references to Awkward records or arrays simply *become* these objects,
which implies some overhead from copying data and a loss of the functionality
that Awkward would bring.

Strings, for instance, are replaced by Numba's built-in string model so that
all string operations will work, but Awkward operations like broadcasting
characters will not.

For this case, the signatures are

```python
# parameters["__record__"] = record_name
ak.behavior["__numba_typer__", record_name] = typer
ak.behavior["__numba_lower__", record_name] = lower

# for an array one-level deep
ak.behavior["__numba_typer__", ".", record_name] = typer
ak.behavior["__numba_lower__", ".", record_name] = lower

# for an array any number of levels deep
ak.behavior["__numba_typer__", "*", record_name] = typer
ak.behavior["__numba_lower__", "*", record_name] = lower

# parameters["__list__"] = list_name
ak.behavior["__numba_typer__", list_name] = typer
ak.behavior["__numba_lower__", list_name] = lower
```

The `typer` function takes an
{func}`ak._connect._numba.arrayview.ArrayViewType` as its only argument
and returns the Numba type of its replacement, while the `lower`
function takes

- `context`: Numba context
- `builder`: Numba builder
- `rettype`: the Numba type of its replacement
- `viewtype`: an {func}`ak._connect._numba.arrayview.ArrayViewType`
- `viewval`: a Numba value of the view
- `viewproxy`: a Numba proxy (`context.make_helper`) of the view
- `attype`: the Numba integer type of the index position
- `atval`: the Numba value of the index position

% Add back once https://github.com/scikit-hep/vector/issues/273 is completed

% Complete example

% ================

% The

% `Vector design prototype <https://vector.readthedocs.io/en/latest/usage/vector_design_prototype.html>`__

% has a complete example, including Numba.

# Axes in Awkward Array

## What is an axis?

An axis describes how operations in Awkward Array traverse, reduce, or
combine elements of a nested array structure.

Like NumPy, axes can be specified positionally using integers. Awkward
extends this concept by allowing axes to be named, which helps preserve
meaning and correctness when working with complex or deeply nested data.

## Positional axes

A positional axis is specified using an integer.

- Positive integers count from the outermost level inward
- Negative integers count from the innermost level outward
- `axis=None` indicates that the operation should be applied across all
  axes of the array

This behavior is consistent with NumPy-style axis semantics, extended to
Awkward’s nested array model.

## Named axes

In addition to positional axes, Awkward supports named axes. A named axis
identifies a dimension by meaning rather than by position.

Named axes are useful when arrays carry metadata describing the role of
different dimensions, allowing operations to remain correct even if the
underlying structure changes.

In the documentation and type annotations, a named axis is represented by
the `AxisName` type.

## Multiple axes

Some operations accept more than one axis at a time. In these cases, axes
may be provided as a tuple.

Each element of the tuple may be a positional or named axis. The order of
axes in the tuple determines the order in which the operation is applied.

In the documentation, this usage is represented by the `AxisTuple` type.

## Axis mappings

Certain operations transform or rename axis metadata rather than operating
on array values directly.

An axis mapping describes how existing axes are renamed or reassigned. This
is primarily used in functions that manipulate named axis metadata, such as
`ak.with_named_axis`.

In the documentation, this behavior is represented by the `AxisMapping`
type.

## Axis types in the API

Types such as `AxisName`, `AxisTuple`, and `AxisMapping` appear in function
signatures and documentation to describe the functional role of the axis
parameter.

They are intended to clarify how an axis is interpreted by an operation,
not to expose internal implementation details.

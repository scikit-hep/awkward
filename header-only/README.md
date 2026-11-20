# Awkward Header-only Libraries

This CMake project defined a set of header-only libraries for working with Awkward Array. Use of these libraries is
described in the [documentation](https://awkward-array.org/doc/main/user-guide/). If you are not using CMake, then
ensure that the contents of `builder-options`, `growable-buffer`, `layout-builder` are on the include path.

On the experimental C++23 branch, LayoutBuilder checks buffer element types with
concepts in `awkward::LayoutBuilder`:

- `Primitive` constrains `Numpy` to supported mutable scalar types: integers up
  to 64 bits, bool, char, float, double, supported long double representations,
  their supported complex counterparts, and `awkward::float16` when available.
- `Index` constrains `ListOffset`, `String`, `Indexed`, and union indices to
  signed 32-bit, unsigned 32-bit, or signed 64-bit integers.
- `SignedIndex` restricts `IndexedOption` to signed 32-bit or 64-bit indices,
  since negative values encode missing entries.
- `UnionTag` restricts union tags to signed 8-bit integers.

Const-qualified, volatile-qualified, and reference buffer element types are
rejected. Unsupported types now fail template constraint checking rather than
producing unsupported forms later. Nested builder interfaces are unchanged.

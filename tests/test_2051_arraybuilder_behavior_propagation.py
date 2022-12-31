import awkward as ak


class Point(ak.Array):
    def length(self):
        raise NotImplementedError


def test():
    behavior = {("*", "point"): Point}
    builder = ak.ArrayBuilder(behavior=behavior)
    with builder.record("point"):
        builder.field("x").real(1.0)
        builder.field("y").real(2.0)
        builder.field("z").real(3.0)

    assert ak._util.arrays_approx_equal(builder, builder.snapshot())

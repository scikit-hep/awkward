# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def _min_pair(array, mask):
    array = ak.typetracer.length_zero_if_typetracer(array)

    # Find location of minimum 0 slot
    i_min = ak.argmin(array["0"], axis=-1, keepdims=True, mask_identity=True)
    # Index into array
    pair_min = ak.ravel(array[ak.from_regular(i_min)], highlevel=False)
    if mask:
        return pair_min
    else:
        form = pair_min.content.form
        length_one_content = form.length_one_array(
            backend=pair_min.backend, highlevel=False
        )
        identity_content = ak.fill_none(length_one_content, 0, highlevel=False)
        identity = ak.record.Record(identity_content, 0)
        return ak.fill_none(pair_min, identity, highlevel=False)


def _argmin_pair(array, mask):
    array = ak.typetracer.length_zero_if_typetracer(array)

    assert not mask
    # Find location of minimum 0 slot
    return ak.argmin(array["0"], axis=-1, keepdims=False, mask_identity=mask)


def _argmin_pair_bad(array, mask):
    array = ak.typetracer.length_zero_if_typetracer(array)

    assert not mask
    # Find location of minimum 0 slot
    return ak.argmin(array["0"], axis=-1, keepdims=False, mask_identity=True)


def test_non_positional():
    behavior = {(ak.min, "pair"): _min_pair}

    x = ak.Array(
        [
            [
                [1, 2, 3],
                [5, 4, 3],
                [2],
            ],
            [
                [8],
                [],
                [10, 4, 4],
            ],
        ]
    )
    y = 2 * x - x**2
    z = ak.zip((x, y), with_name="pair", behavior=behavior)

    assert ak.almost_equal(
        ak.min(z, axis=-1, mask_identity=True),
        ak.Array(
            [
                [(1, 1), (3, -3), (2, 0)],
                [
                    (8, -48),
                    None,
                    (4, -8),
                ],
            ],
            with_name="pair",
        ),
    )

    assert ak.almost_equal(
        ak.min(z, axis=-1, mask_identity=False),
        ak.Array(
            [
                [(1, 1), (3, -3), (2, 0)],
                [
                    (8, -48),
                    (0, 0),
                    (4, -8),
                ],
            ],
            with_name="pair",
        ),
    )


def test_positional_bad():
    behavior = {(ak.argmin, "pair"): _argmin_pair_bad}

    x = ak.Array(
        [
            [
                [1, 2, 3],
                [5, 4, 3],
                [2],
            ],
            [
                [8],
                [],
                [10, 4, 4],
            ],
        ]
    )
    y = 2 * x - x**2
    z = ak.zip((x, y), with_name="pair", behavior=behavior)

    with pytest.raises(TypeError, match=r"'pair' returned an option"):
        assert ak.almost_equal(
            ak.argmin(z, axis=-1, mask_identity=True), [[0, 2, 0], [0, None, 1]]
        )

    with pytest.raises(TypeError, match=r"'pair' returned an option"):
        assert ak.almost_equal(
            ak.argmin(z, axis=-1, mask_identity=False), [[0, 2, 0], [0, -1, 1]]
        )


def test_positional_good():
    behavior = {(ak.argmin, "pair"): _argmin_pair}

    x = ak.Array(
        [
            [
                [1, 2, 3],
                [5, 4, 3],
                [2],
            ],
            [
                [8],
                [],
                [10, 4, 4],
            ],
        ]
    )
    y = 2 * x - x**2
    z = ak.zip((x, y), with_name="pair", behavior=behavior)

    assert ak.almost_equal(
        ak.argmin(z, axis=-1, mask_identity=True), [[0, 2, 0], [0, None, 1]]
    )

    assert ak.almost_equal(
        ak.argmin(z, axis=-1, mask_identity=False), [[0, 2, 0], [0, -1, 1]]
    )

import numpy as np

from quinkgl import serialization as serialization_pkg
from quinkgl.serialization import deserialize as pkg_deserialize
from quinkgl.serialization import serialize as pkg_serialize
from quinkgl.serialization.weights import deserialize, serialize
from quinkgl.utils.serialization import deserialize as legacy_deserialize
from quinkgl.utils.serialization import serialize as legacy_serialize


def test_weight_serialization_round_trip():
    data = [np.array([1, 2, 3]), np.array([[4.0, 5.0]])]
    restored = deserialize(serialize(data))
    assert len(restored) == 2
    assert np.array_equal(restored[0], data[0])
    assert np.array_equal(restored[1], data[1])


def test_serialization_import_surfaces_reexport_weight_helpers():
    assert serialization_pkg.serialize is pkg_serialize is serialize
    assert serialization_pkg.deserialize is pkg_deserialize is deserialize
    assert legacy_serialize is serialize
    assert legacy_deserialize is deserialize


def test_weight_serialization_preserves_order_for_eleven_plus_arrays():
    data = [np.array([index]) for index in range(12)]

    restored = deserialize(serialize(data))

    assert len(restored) == len(data)
    for index, array in enumerate(restored):
        assert np.array_equal(array, data[index])

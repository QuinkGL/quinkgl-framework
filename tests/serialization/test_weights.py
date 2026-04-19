import warnings

import numpy as np
import pytest

from quinkgl import serialization as serialization_pkg
from quinkgl.serialization import deserialize as pkg_deserialize
from quinkgl.serialization import serialize as pkg_serialize
from quinkgl.serialization.weights import (
    deserialize,
    deserialize_numpy_weights,
    serialize,
    serialize_numpy_weights,
)
from quinkgl.utils.serialization import deserialize as legacy_deserialize
from quinkgl.utils.serialization import serialize as legacy_serialize


def test_weight_serialization_round_trip():
    data = [np.array([1, 2, 3]), np.array([[4.0, 5.0]])]
    restored = deserialize_numpy_weights(serialize_numpy_weights(data))
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

    restored = deserialize_numpy_weights(serialize_numpy_weights(data))

    assert len(restored) == len(data)
    for index, array in enumerate(restored):
        assert np.array_equal(array, data[index])


def test_deprecated_aliases_emit_deprecation_warning():
    data = [np.array([1, 2, 3])]
    with pytest.warns(DeprecationWarning, match="serialize_numpy_weights"):
        payload = serialize(data)
    with pytest.warns(DeprecationWarning, match="deserialize_numpy_weights"):
        restored = deserialize(payload)
    assert np.array_equal(restored[0], data[0])


def test_non_deprecated_path_emits_no_warning():
    data = [np.array([1, 2, 3])]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        deserialize_numpy_weights(serialize_numpy_weights(data))

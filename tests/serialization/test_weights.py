import warnings

import numpy as np
import pytest

from quinkgl.serialization.weights import (
    deserialize_numpy_weights,
    serialize_numpy_weights,
)


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


def test_ssert np.array_equal(array, data[index])


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

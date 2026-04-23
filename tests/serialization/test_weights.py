import warnings

import numpy as np
import pytest

from quinkgl.serialization.weights import (
    deserialize_numpy_weights,
    serialize_numpy_weights,
    serialize,
    deserialize,
)


def test_weight_serialization_round_trip():
    data = [np.array([1, 2, 3]), np.array([[4.0, 5.0]])]
    restored = deserialize_numpy_weights(serialize_numpy_weights(data))
    assert len(restored) == 2
    assert np.array_equal(restored[0], data[0])
    assert np.array_equal(restored[1], data[1])


def test_serialization_import_surfaces_reexport_weight_helpers():
    import quinkgl.serialization as serialization_pkg
    from quinkgl.serialization.weights import serialize as weights_serialize
    from quinkgl.serialization.weights import deserialize as weights_deserialize
    assert serialization_pkg.serialize is weights_serialize
    assert serialization_pkg.deserialize is weights_deserialize


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

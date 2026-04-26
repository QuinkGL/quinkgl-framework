from importlib.metadata import PackageNotFoundError, version

import quinkgl


def test_version_is_set():
    assert quinkgl.__version__ is not None
    assert isinstance(quinkgl.__version__, str)
    assert len(quinkgl.__version__) > 0


def test_version_matches_distribution_metadata_when_available():
    try:
        expected_version = version("quinkgl")
    except PackageNotFoundError:
        expected_version = "0.3.3"

    assert quinkgl.__version__ == expected_version


def test_missing_data_symbols_are_not_exported():
    assert "DatasetLoader" not in quinkgl.__all__
    assert "FederatedDataSplitter" not in quinkgl.__all__
    assert "DatasetInfo" not in quinkgl.__all__
    assert not hasattr(quinkgl, "DatasetLoader")
    assert not hasattr(quinkgl, "FederatedDataSplitter")
    assert not hasattr(quinkgl, "DatasetInfo")


def test_tensorflow_symbol_export_tracks_availability():
    if hasattr(quinkgl, "TensorFlowModel"):
        assert "TensorFlowModel" in quinkgl.__all__
    else:
        assert "TensorFlowModel" not in quinkgl.__all__


def test_all_exported_names_are_importable():
    """T3: Verify every name in quinkgl.__all__ can be imported without error."""
    for name in quinkgl.__all__:
        # Skip module-level attributes that aren't actually imports
        if name.startswith("_"):
            continue
        # Try to get the attribute from quinkgl
        assert hasattr(quinkgl, name), f"{name} is in __all__ but not accessible on quinkgl module"
        attr = getattr(quinkgl, name)
        # If it's None (e.g., TensorFlowModel when TF not installed), that's okay
        # as long as the attribute exists
        if attr is not None or name == "_tensorflow_available" or name == "_data_available":
            continue

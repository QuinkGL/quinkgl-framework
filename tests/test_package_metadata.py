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
        expected_version = "0.3.1"

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

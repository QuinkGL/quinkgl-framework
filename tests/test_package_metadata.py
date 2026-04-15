import quinkgl


def test_version_is_set():
    assert quinkgl.__version__ is not None
    assert isinstance(quinkgl.__version__, str)
    assert len(quinkgl.__version__) > 0

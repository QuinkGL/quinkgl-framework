from quinkgl.aggregation import FedAvg, FedAvgM, FedProx, Krum, MultiKrum, TrimmedMean


def test_aggregation_symbols_are_importable():
    assert FedAvg is not None
    assert FedProx is not None
    assert FedAvgM is not None
    assert TrimmedMean is not None
    assert Krum is not None
    assert MultiKrum is not None

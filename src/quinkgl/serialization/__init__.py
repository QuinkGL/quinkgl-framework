"""Serialization helpers for QuinkGL model weights."""

from quinkgl.serialization.weights import (
    serialize_numpy_weights,
    deserialize_numpy_weights,
    serialize,   # deprecated alias
    deserialize, # deprecated alias
)
from quinkgl.serialization.quantization import (
    QuantizationConfig,
    quantize_weights,
    dequantize_weights,
)
from quinkgl.serialization.sparsification import (
    SparsificationConfig,
    DeltaCompressionConfig,
    sparsify_weights,
    desparsify_weights,
    compute_delta,
    apply_delta,
)
from quinkgl.serialization.compression import (
    CompressionConfig,
    compress_weights,
    decompress_weights,
    compress_decompress_roundtrip,
)
from quinkgl.serialization.error_feedback import (
    ErrorFeedbackState,
    ErrorFeedbackConfig,
)

__all__ = [
    "serialize_numpy_weights", "deserialize_numpy_weights",
    "serialize", "deserialize",  # deprecated aliases
    "QuantizationConfig", "quantize_weights", "dequantize_weights",
    "SparsificationConfig", "DeltaCompressionConfig",
    "sparsify_weights", "desparsify_weights",
    "compute_delta", "apply_delta",
    "CompressionConfig", "compress_weights", "decompress_weights",
    "compress_decompress_roundtrip",
    "ErrorFeedbackState", "ErrorFeedbackConfig",
]

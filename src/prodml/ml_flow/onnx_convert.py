"""Convert a trained model to ONNX. One function per source framework."""

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def _from_sklearn(model, n_features: int) -> bytes:
    from skl2onnx import to_onnx

    dummy = np.zeros((1, n_features), dtype=np.float32)
    # zipmap=False makes probabilities a plain (N, 2) array instead of
    # a list of dicts, which is far easier to consume at inference time.
    onx = to_onnx(model, dummy, options={id(model): {"zipmap": False}})
    return onx.SerializeToString()


def _from_xgboost(model, n_features: int) -> bytes:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onx = convert_xgboost(model, initial_types=initial_type)
    return onx.SerializeToString()


def _from_torch(model, n_features: int) -> bytes:
    import io

    import torch
    from torch import nn

    # The MLP outputs raw logits, because BCEWithLogitsLoss wants them.
    # Append a Sigmoid so the exported graph emits probabilities directly —
    # otherwise every consumer has to remember to apply it.
    exportable = nn.Sequential(model, nn.Sigmoid())
    exportable.eval()

    dummy = torch.zeros(1, n_features, dtype=torch.float32)
    buf = io.BytesIO()
    torch.onnx.export(
        exportable,
        dummy,
        buf,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
    )
    return buf.getvalue()


CONVERTERS = {
    "logistic_regression": _from_sklearn,
    "xgboost": _from_xgboost,
    "pytorch": _from_torch,
}


def convert(model, framework: str, n_features: int) -> bytes:
    """Serialised ONNX graph for a trained model."""
    if framework not in CONVERTERS:
        raise ValueError(f"No ONNX converter for framework {framework!r}")
    onnx_bytes = CONVERTERS[framework](model, n_features)
    log.info("onnx_converted", framework=framework, n_features=n_features,
             size_bytes=len(onnx_bytes))
    return onnx_bytes
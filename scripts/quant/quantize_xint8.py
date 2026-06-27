"""
Low-memory XINT8 quantization for Laneformer ONNX with AMD Quark 0.11.2.

This intentionally bypasses ModelQuantizer.quantize_model. In Quark 0.11.2 that
entrypoint eagerly onnx.load()s external weights and then save/reload shape-infers
the full model before any calibration log is printed. For this 4.4 GB FP16 model
that creates avoidable full-model copies.

The path here is:
1. load only the ONNX header (external weights stay on disk);
2. calibrate from a header-only augmented model, with hardlinks to the original
   external .data file in Quark temp dirs;
3. run Quark's XINT8QDQQuantizer directly, while monkey-patching Quark's weight
   tensor reader to memmap one external tensor at a time;
4. stream the quantized model writeout to an external .data file.
"""

from __future__ import annotations

import contextlib
import copy
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, external_data_helper
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.quant_utils import add_infer_metadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")

from quark.onnx import XINT8_QCONFIG  # noqa: E402
from quark.onnx.calibration import load_tensors_range, run_calibration, save_tensors_range  # noqa: E402
from quark.onnx.calibration import calibrators as quark_calibrators  # noqa: E402
from quark.onnx.calibration import calibrate as quark_calibrate_mod  # noqa: E402
from quark.onnx.quantization.config.config import QConfig  # noqa: E402
from quark.onnx.quantization.config.maps import _check_q_config, _map_q_config  # noqa: E402
from quark.onnx.quantization.quant_utils import (  # noqa: E402
    check_model_quantizable,
    get_all_target_nodes,
    get_exclude_nodes,
    get_matmul_nodes_without_weights,
)
from quark.onnx.quantizers import get_static_op_types, run_static_quantization  # noqa: E402
from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model  # noqa: E402
from quark.onnx.utils.system_utils import update_tmp_dir  # noqa: E402

import quark.onnx.quantizers.onnx_quantizer as quark_onnx_quantizer  # noqa: E402
import quark.onnx.quantizers.qdq_quantizer as quark_qdq_quantizer  # noqa: E402


SRC = Path(os.environ.get("QUARK_XINT8_SRC", "/home/antoinefa/kog/onnx/out_fp16/laneformer.onnx"))
DST = Path(os.environ.get("QUARK_XINT8_DST", "/home/antoinefa/kog/quant/out_xint8/laneformer_xint8.onnx"))
TMP = Path(os.environ.get("QUARK_XINT8_TMP", "/home/antoinefa/kog/quant/tmp"))
RANGES = Path(os.environ.get("QUARK_XINT8_RANGES", TMP / "laneformer_xint8_ranges.json"))


class TokenCalibReader(CalibrationDataReader):
    def __init__(self, n: int = 8, seqlen: int = 64, vocab: int = 32000):
        rng = np.random.default_rng(0)
        self.samples = [
            {"input_ids": rng.integers(1, vocab, size=(1, seqlen), dtype=np.int64)}
            for _ in range(n)
        ]
        self._it = iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def reset_iter(self) -> None:
        self._it = iter(self.samples)

    def get_next(self):
        return next(self._it, None)


class PeakRSS:
    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._peak = 0
        self.peak_gb = 0.0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        print(f"[mem] start {self.label}: rss={_rss_gb():.2f} GB avail={_mem_available_gb():.2f} GB")
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.peak_gb = self._peak / 1024**3
        print(f"[mem] peak {self.label}: {self.peak_gb:.2f} GB (process max {ru / 1024**3:.2f} GB)")
        return False

    def _sample(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, _rss_bytes())
            time.sleep(0.05)


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _rss_gb() -> float:
    return _rss_bytes() / 1024**3


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


def _load_header_model() -> onnx.ModelProto:
    model = onnx.load_model(SRC.as_posix(), load_external_data=False)
    add_infer_metadata(model)
    return model


def _external_locations(model: onnx.ModelProto) -> set[str]:
    locations: set[str] = set()
    for tensor in external_data_helper._get_all_tensors(model):  # type: ignore[attr-defined]
        if external_data_helper.uses_external_data(tensor):
            info = external_data_helper.ExternalDataInfo(tensor)
            if info.location:
                locations.add(info.location)
    return locations


def _resolve_external_path(location: str, base_dir: Path = SRC.parent) -> Path:
    path = Path(location)
    if path.is_absolute():
        return path
    return base_dir / path


def _link_external_data(temp_dir: Path, locations: set[str]) -> None:
    for location in locations:
        src = _resolve_external_path(location)
        dst = temp_dir / location
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        os.link(src, dst)


@contextlib.contextmanager
def _patched_calibration_tempdirs(model: onnx.ModelProto) -> Iterator[None]:
    locations = _external_locations(model)
    original = quark_calibrate_mod.create_tmp_dir

    def create_tmp_dir_with_data(prefix: str):
        td = original(prefix)
        _link_external_data(Path(td.name), locations)
        return td

    quark_calibrate_mod.create_tmp_dir = create_tmp_dir_with_data
    try:
        yield
    finally:
        quark_calibrate_mod.create_tmp_dir = original


@contextlib.contextmanager
def _patched_pow2_calibration_session() -> Iterator[None]:
    original = quark_calibrators.PowOfTwoCalibrater.create_inference_session

    def create_inference_session_low_mem(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        if self.use_external_data_format:
            self.infer_session = create_infer_session_for_onnx_model(
                self.augmented_model_path,
                sess_options=sess_options,
                providers=self.execution_providers,
                use_external_data_format=self.use_external_data_format,
            )
        else:
            self.infer_session = create_infer_session_for_onnx_model(
                self.model,
                sess_options=sess_options,
                providers=self.execution_providers,
            )

    quark_calibrators.PowOfTwoCalibrater.create_inference_session = create_inference_session_low_mem
    try:
        yield
    finally:
        quark_calibrators.PowOfTwoCalibrater.create_inference_session = original


def _external_tensor_to_array(tensor: TensorProto) -> np.ndarray:
    if tensor.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        raise ValueError(f"Only float type is supported. Weights {tensor.name} is {tensor.data_type}")

    if not external_data_helper.uses_external_data(tensor):
        return onnx.numpy_helper.to_array(tensor)

    info = external_data_helper.ExternalDataInfo(tensor)
    if info.length is None:
        raise ValueError(f"External tensor {tensor.name!r} is missing a length field.")

    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    shape = tuple(int(d) for d in tensor.dims)
    return np.memmap(
        _resolve_external_path(info.location),
        dtype=dtype,
        mode="r",
        offset=info.offset or 0,
        shape=shape,
        order="C",
    )


@contextlib.contextmanager
def _patched_streaming_weight_reader() -> Iterator[None]:
    original_qdq = quark_qdq_quantizer.tensor_proto_to_array
    original_onnx = quark_onnx_quantizer.tensor_proto_to_array
    quark_qdq_quantizer.tensor_proto_to_array = _external_tensor_to_array
    quark_onnx_quantizer.tensor_proto_to_array = _external_tensor_to_array
    try:
        yield
    finally:
        quark_qdq_quantizer.tensor_proto_to_array = original_qdq
        quark_onnx_quantizer.tensor_proto_to_array = original_onnx


def _set_external_data(tensor: TensorProto, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", offset), ("length", length)):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = str(value)


def _copy_file_slice(src: Path, dst_f, offset: int, length: int) -> None:
    remaining = length
    with open(src, "rb") as src_f:
        src_f.seek(offset)
        while remaining:
            chunk = src_f.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise IOError(f"Unexpected EOF while copying external data from {src}")
            dst_f.write(chunk)
            remaining -= len(chunk)


def _save_external_streaming(model: onnx.ModelProto, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data_name = output.name + ".data"
    data_path = output.parent / data_name
    tmp_data_path = data_path.with_suffix(data_path.suffix + ".tmp")
    if tmp_data_path.exists():
        tmp_data_path.unlink()
    if data_path.exists():
        data_path.unlink()

    offset = 0
    with open(tmp_data_path, "wb") as data_f:
        for tensor in external_data_helper._get_all_tensors(model):  # type: ignore[attr-defined]
            if tensor.HasField("raw_data") and len(tensor.raw_data) > 0:
                raw = tensor.raw_data
                data_f.write(raw)
                length = len(raw)
                tensor.ClearField("raw_data")
                _set_external_data(tensor, data_name, offset, length)
                offset += length
            elif external_data_helper.uses_external_data(tensor):
                info = external_data_helper.ExternalDataInfo(tensor)
                src = _resolve_external_path(info.location)
                start = info.offset or 0
                length = info.length if info.length is not None else src.stat().st_size - start
                _copy_file_slice(src, data_f, start, length)
                _set_external_data(tensor, data_name, offset, length)
                offset += length

    os.replace(tmp_data_path, data_path)
    onnx.save_model(model, output.as_posix())


def _convert_qdq_scales_to_fp32(model: onnx.ModelProto) -> None:
    scale_names = {
        node.input[1]
        for node in model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear", "ExtendedQuantizeLinear", "ExtendedDequantizeLinear"}
        and len(node.input) > 1
    }
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name not in scale_names or tensor.data_type != TensorProto.FLOAT16:
            continue
        scale = onnx.numpy_helper.to_array(tensor).astype(np.float32)
        model.graph.initializer[index].CopyFrom(onnx.numpy_helper.from_array(scale, tensor.name))


def _build_mapping() -> dict:
    cfg = QConfig(
        global_config=copy.deepcopy(XINT8_QCONFIG.global_config),
        use_external_data_format=True,
        # Kept for source traceability. This script bypasses apply_pre_process.
        SkipPreprocess=True,
        OptimizeModel=False,
        SimplifyModel=False,
        QuantizeFP16=True,
        UseFP32Scale=True,
        ConvertFP16ToFP32=False,
        CalibOptimizeMem=True,
        CalibWorkerNum=1,
        MinMSEModePof2Scale="MostCommon",
        TmpDir=TMP.as_posix(),
        PrintSummary=False,
    )
    _check_q_config(cfg)
    mapping = _map_q_config(cfg, SRC.as_posix())
    mapping["extra_options"]["SkipPreprocess"] = True
    mapping["extra_options"]["OptimizeModel"] = False
    mapping["extra_options"]["SimplifyModel"] = False
    mapping["extra_options"]["QuantizeFP16"] = True
    mapping["extra_options"]["UseFP32Scale"] = True
    mapping["extra_options"]["ConvertFP16ToFP32"] = False
    mapping["extra_options"]["CalibOptimizeMem"] = True
    mapping["extra_options"]["CalibWorkerNum"] = 1
    mapping["extra_options"]["MinMSEModePof2Scale"] = "MostCommon"
    mapping["extra_options"]["TmpDir"] = TMP.as_posix()
    mapping["extra_options"]["PrintSummary"] = False
    return mapping


def _prepare_quant_args(model: onnx.ModelProto, mapping: dict) -> dict:
    extra = mapping["extra_options"]
    enable_npu_cnn = extra["EnableNPUCnn"]
    enable_npu_transformer = extra.get("EnableNPUTransformer", False)
    quant_format = mapping["quant_format"]

    nodes_to_exclude = get_all_target_nodes(model, mapping["nodes_to_exclude"] + mapping["subgraphs_to_exclude"])
    input_nodes = extra["InputNodes"]
    output_nodes = extra["OutputNodes"]
    if input_nodes or output_nodes:
        nodes_to_exclude += get_exclude_nodes(model, input_nodes, output_nodes)

    if extra.get("MatMulConstBOnly", enable_npu_transformer):
        nodes_to_exclude += get_matmul_nodes_without_weights(model)

    if enable_npu_cnn or enable_npu_transformer:
        extra.setdefault("ConvertSplitToSlice", True)
        extra.setdefault("ConvertBNToConv", True)
        extra.setdefault("ConvertReduceMeanToGlobalAvgPool", True)
        extra.setdefault("SplitLargeKernelPool", True)

    op_types_to_quantize = get_static_op_types(
        model,
        extra["OpTypesToQuantize"],
        extra["ExtraOpTypesToQuantize"],
        enable_npu_cnn,
        enable_npu_transformer,
        quant_format,
        extra,
    )
    if not check_model_quantizable(model, op_types_to_quantize, nodes_to_exclude):
        raise RuntimeError("No quantizable activation tensors found in the header model.")

    return {
        "op_types_to_quantize": op_types_to_quantize,
        "nodes_to_quantize": extra["NodesToQuantize"],
        "nodes_to_exclude": nodes_to_exclude,
        "enable_npu_cnn": enable_npu_cnn,
        "enable_npu_transformer": enable_npu_transformer,
    }


def _calibrate_if_needed(model: onnx.ModelProto, mapping: dict, qargs: dict) -> None:
    if RANGES.exists() and os.environ.get("QUARK_XINT8_RECALIBRATE") != "1":
        print(f"[quark] reuse calibration ranges: {RANGES}")
        return

    print(f"[quark] calibrating to {RANGES}")
    with PeakRSS("calibration"), _patched_calibration_tempdirs(model), _patched_pow2_calibration_session():
        ranges = run_calibration(
            model,
            TokenCalibReader(),
            qargs["op_types_to_quantize"],
            mapping["activation_type"].map_onnx_format,
            mapping["calibrate_method"],
            mapping["use_external_data_format"],
            mapping["extra_options"]["ExecutionProviders"],
            mapping["extra_options"],
        )
    RANGES.parent.mkdir(parents=True, exist_ok=True)
    save_tensors_range(ranges, RANGES.as_posix())


def _quantize(model: onnx.ModelProto, mapping: dict, qargs: dict) -> onnx.ModelProto:
    ranges = load_tensors_range(RANGES.as_posix())
    if ranges is None:
        raise RuntimeError(f"Failed to load calibration ranges from {RANGES}")

    print(f"[quark] static XINT8 quantization to {DST}")
    with PeakRSS("quantization"), _patched_streaming_weight_reader():
        quant_model = run_static_quantization(
            model,
            ranges,
            mapping["per_channel"],
            False,
            mapping["weight_type"].map_onnx_format,
            mapping["activation_type"].map_onnx_format,
            qargs["enable_npu_cnn"],
            qargs["enable_npu_transformer"],
            mapping["quant_format"],
            mapping["calibrate_method"],
            qargs["nodes_to_quantize"],
            qargs["nodes_to_exclude"],
            qargs["op_types_to_quantize"],
            mapping["extra_options"],
        )
        _convert_qdq_scales_to_fp32(quant_model)
        return quant_model


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    DST.parent.mkdir(parents=True, exist_ok=True)
    update_tmp_dir(TMP.as_posix())

    print(f"[quark] source header={SRC} data={SRC.with_name(SRC.name + '.data')}")
    print(f"[quark] available memory before start: {_mem_available_gb():.2f} GB")

    mapping = _build_mapping()
    with PeakRSS("header load"):
        model = _load_header_model()
    qargs = _prepare_quant_args(model, mapping)

    _calibrate_if_needed(model, mapping, qargs)

    # Reload a clean header for quantization because calibration augments its copy.
    with PeakRSS("header reload"):
        model = _load_header_model()
    qargs = _prepare_quant_args(model, mapping)
    quant_model = _quantize(model, mapping, qargs)

    with PeakRSS("streaming save"):
        _save_external_streaming(quant_model, DST)

    data_path = DST.with_name(DST.name + ".data")
    total_gb = (DST.stat().st_size + data_path.stat().st_size) / 1024**3
    print(f"[quark] wrote {DST} + {data_path.name}: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()

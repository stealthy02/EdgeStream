# 300-frame benchmark artifacts

The committed 300-frame summaries are:

- `../onnx/orangepi_infer_stats.json` (ONNX Runtime)
- `../rknn/orangepi_infer_stats.json` (RKNN FP16)
- `../rknn/orangepi_int8_infer_stats.json` (RKNN INT8)

All three contain `frame_count=300`, mean and sample standard deviation in milliseconds. RKNN FP16 and INT8 runs retain per-frame raw timing at `../rknn/orangepi_300f_raw.jsonl` and `../rknn/orangepi_int8_300f_raw.jsonl`; the ONNX summary retains aggregate statistics only. The cross-run interpretation is in [`performance_analysis.md`](performance_analysis.md).

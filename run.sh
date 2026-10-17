#!/usr/bin/env bash
# SPECTRA entrypoint: video -> two-frame VGGT atlas -> TensorRT VGGT -> pose/depth decode
# -> Open3D filtering -> cloud.ply + metadata.json + events.jsonl (geometry_ready).
# Model bundle: models/vggt-onnx/vggt.onnx (+ vggt.onnx.data). Engine is built from it.
set -euo pipefail
cd "$(dirname "$0")"
VIDEO=${1:-/home/atharv/spectra/input/kitchen-sample.mp4}
OUTPUT=${2:-/home/atharv/spectra/output}
BUNDLE=${BUNDLE:-models/vggt-onnx}
ENGINE=${ENGINE:-$BUNDLE/vggt.engine}
python3 -m spectra.deployment.reference "$VIDEO" --backend tensorrt --device cuda \
  --bundle "$BUNDLE" --engine "$ENGINE" --output "$OUTPUT"

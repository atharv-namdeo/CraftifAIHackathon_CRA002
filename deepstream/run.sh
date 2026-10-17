#!/usr/bin/env bash
# SPECTRA on DeepStream: the PipeGen-generated pipeline (PipeGen v0.5.1, session
# 0707a4b2) with its two VGGT stages completed in parsers/vggt_spectra.cpp.
#   video -> nvstreammux 1036x518 -> VggtPair [frame t-1s | frame t]
#         -> nvinfer_backport (PipeGen-built FP16 TensorRT VGGT engine)
#         -> VggtDecode (cameras, world points, filtering, PLY/metadata/events)
#         -> meta_json (events.json) -> annotated JPEG frames
# Runs inside the PipeGen DeepStream container (craftifai-amd64-v3).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VIDEO="${VIDEO:-/home/atharv/spectra/input/kitchen-sample.mp4}"
OUT="${OUT:-/home/atharv/spectra/output}"
export VGGT_OUT="$OUT"
export VGGT_ENGINE="${VGGT_ENGINE:-$(sed -n 's/^model-engine-file=//p' "$DIR/configs/config_vggt.txt")}"
export VGGT_SRC_W="${VGGT_SRC_W:-778}" VGGT_SRC_H="${VGGT_SRC_H:-520}" VGGT_FPS="${VGGT_FPS:-4}"
export VGGT_SAMPLE_SECONDS="${VGGT_SAMPLE_SECONDS:-1.0}"
export JSON_OUT="$OUT/events.json" JSON_SOURCE="$VIDEO" JSON_MIN_CONF=0.0 JSON_MASK_OUTLINE=0
mkdir -p "$OUT/frames" "$OUT/pairs"
rm -f "$OUT"/frames/frame_*.jpg "$OUT"/pairs/pair_* "$OUT/cloud.ply" "$OUT/metadata.json"

make -C "$DIR/parsers"
LIB="$DIR/build/libvggt_spectra.so"

gst-launch-1.0 -e \
  nvstreammux name=mux batch-size=1 width=1036 height=518 batched-push-timeout=40000 live-source=0 \
  nvurisrcbin uri="file://$VIDEO" ! mux.sink_0 \
  mux. ! queue max-size-buffers=4 ! nvvideoconvert \
  ! nvcustomprocess custom-lib-path="$LIB" process-func-name=VggtPair \
  ! queue max-size-buffers=4 ! nvinfer_backport config-file-path="$DIR/configs/config_vggt.txt" \
  ! queue max-size-buffers=4 ! nvcustomprocess custom-lib-path="$LIB" process-func-name=VggtDecode \
  ! queue max-size-buffers=4 ! nvcustomprocess custom-lib-path="$DIR/build/libmeta_json.so" process-func-name=MetaJson \
  ! queue max-size-buffers=4 ! nvvideoconvert ! "video/x-raw,format=I420" ! jpegenc \
  ! multifilesink location="$OUT/frames/frame_%05d.jpg"

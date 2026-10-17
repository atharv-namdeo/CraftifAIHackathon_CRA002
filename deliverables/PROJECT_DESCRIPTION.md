# SPECTRA — VGGT-only geometric mapping via CraftifAI PipeGen

**Track:** A — PipeGen. **Model:** VGGT (the only neural network).

## Problem
Recover usable 3D geometry (camera poses + a point cloud) from an ordinary short video,
deployed as an optimized NVIDIA pipeline instead of a research notebook.

## Inputs
A short local video of a static scene with camera motion. Two frames one second apart are
sampled, preprocessed with upstream VGGT `load_and_preprocess_images(mode="pad")` to 518x518
each, and packed side by side into one RGB atlas: `images`, FP32, NCHW `[1,3,518,1036]`, [0,1].
Bundled test input: `vggt/examples/kitchen/images/00.png` + `01.png` (real upstream images);
`scripts/make_sample_video.py` builds a short real-scene video from the same kitchen set.

## Pipeline tasks
video decode → two-frame temporal sampling → VGGT pad preprocessing → RGB atlas packing →
TensorRT VGGT inference → raw tensor extraction → pose decoding + depth unprojection →
confidence / voxel / outlier filtering (Open3D in Python, a C++ port in DeepStream) → PLY + JSON sinks (`geometry_ready` / `inspect_reconstruction` event).

## Model / checkpoint
Official `facebook/VGGT-1B`, `model.safetensors`, SHA256
`f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` (verified before load by
`scripts/fetch_checkpoint.py`). Upstream VGGT pinned at commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`, unmodified. Full aggregator backbone + camera and
depth heads on exactly two views; point-map and tracking heads disabled via upstream options.

## Outputs
| Name | Shape | Meaning |
|---|---|---|
| `pose_enc` | [1,2,9] | translation, XYZW quaternion, field of view |
| `depth` | [1,2,518,518] | depth, arbitrary scale |
| `depth_conf` | [1,2,518,518] | VGGT confidence |

Actionable output: `cloud.ply`, `metadata.json` (cameras, bounds, filters, engine path) and
`events.jsonl` with one `geometry_ready` / `inspect_reconstruction` event per frame pair (point
count, PLY and metadata paths). Scale is arbitrary (never labelled metres).

## Precision
TensorRT FP16 engine with FP32 input/output, built by PipeGen v0.5.1 (TensorRT 10.3.0) from the
`tensorrt_prep`-processed ONNX bundle: 2,397.4 MB, 1,336 layers (1,274 FP16 / 56 FP32 / 6 other),
build 506.7 s, SHA256 `c52aa61c79979cb4f31e75e4d7288efffae0f493116ae9d4e271aa8765e9476a`.
FP32 weights (4.6 GB) cannot fit the 4 GB GPU.

## Hardware actually available
NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB, driver 595.76, Windows 11 + WSL2 (Ubuntu 22.04), 32 GB
RAM, PipeGen v0.5.1 container `craftifaidocker/deepstream:7.1-amd64-v3`. The intended RTX 5060
target is not present on this machine.

## Status — PipeGen stages
| Stage | Result | Evidence |
|---|---|---|
| Requirement | 13/13 checks; local source repo, `bash run.sh`, model `vggt.onnx` | `evidence/pipegen/07_*.png`, `raw/RequirementState.json` |
| Pipeline IR | Generated and approved | `evidence/pipeline_ir.png`, `raw/PipelineIR.json` |
| Model | ONNX inspection 0 unresolved; FP16 engine built; benchmark **1,464 ms/frame, 57.5 W**; accuracy step unavailable for a "custom" task | `pipegen/12_*.png`, `tensorrt_speed_report.png`, `pipegen/14_*.png` |
| Compiler | DeepStream pipeline generated, `compile=ok`; decode stage left as a pass-through stub by PipeGen's repair loop | `pipegen/15_*.png`, `pipegen/generated_pipeline/` |
| Run | PipeGen's 5 s smoke test fails (engine load alone ~9 s). Full run of the unmodified pipeline: 8 frames, exit 0, no geometry. Run with the two VGGT stages completed in `deepstream/parsers/vggt_spectra.cpp`: 8 frames, 4 frame pairs → 4 verified PLYs (41,720–72,551 points), `geometry_ready` events | `pipegen/run1_generated_passthrough/`, `deepstream_run/` |

Workarounds needed to get VGGT-1B through PipeGen: WSL memory 26 GB + swap (inspection OOM),
`spectra/deployment/tensorrt_prep.py` (constant folding, slice fixes, INT64 inlining; ONNX parity
unchanged), a symlink so PipeGen's compiler finds the external weights, and an explicit FP16
request.

## Measured results
| Measurement | Value |
|---|---|
| PipeGen TensorRT benchmark (FP16, GPU) | 1,464.4 ms / frame, ~1 fps, 57.53 W |
| ONNX Runtime CPU forward pass (same model, FP32) | 37.7 s |
| DeepStream pipeline wall time | ~27 s / frame (unmodified PipeGen pipeline equally slow; cause not isolated, suspected 4 GB VRAM pressure under WSL2) |
| Points per frame pair (after filtering) | 50,494 / 51,976 / 72,551 / 41,720 |
| Agreement with ONNX CPU reference, frames 0+4 | camera baseline −0.5%, view-1 rotation 1.27°, points 50,494 vs 51,678 |

## Limitations
Exactly two views per reconstruction; arbitrary scale; pair clouds are not merged; DeepStream
throughput far below the engine benchmark; PipeGen labels the benchmark input 1×3×518×518 although
the engine input is 1×3×518×1036; one indoor test scene.

## Bonus target
None claimed.

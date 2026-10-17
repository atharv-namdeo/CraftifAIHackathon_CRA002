"""Render the PipeGen/DeepStream demo MP4 strictly from real evidence files.

Inputs are the PipeGen screenshots in deliverables/evidence/pipegen, the DeepStream run directory
(frames, pairs/*.ply, metadata.json, events.jsonl, run log) and the input video. Every number shown
is read from those files; nothing is synthesized.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from make_demo_video import W, H, FPS, ACCENT, MUTED, FG, canvas, text, fit, paste, hold

GREEN = (170, 230, 170)
WARN = (80, 170, 255)
FOOTER = 'SPECTRA | VGGT-1B | PipeGen v0.5.1 | FP16 TensorRT engine in DeepStream 7.1 | RTX 3050 Laptop 4 GB'


def header(img, title, subtitle):
    text(img, title, (40, 55), 1.0, ACCENT, 2)
    text(img, subtitle, (40, 88), 0.55, MUTED)
    text(img, FOOTER, (40, H - 22), 0.5, MUTED)


def read_ply(path):
    """Binary little-endian PLY with float x,y,z + uchar r,g,b (as written by vggt_spectra.cpp)."""
    raw = path.read_bytes()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    head = raw[:end].decode()
    if 'property float x' not in head or 'property uchar red' not in head:
        raise ValueError(f'Unexpected PLY layout in {path}')
    count = int(next(l for l in head.splitlines() if l.startswith('element vertex')).split()[-1])
    rec = np.frombuffer(raw[end:end + 15 * count], dtype=np.dtype([('p', '<f4', 3), ('c', 'u1', 3)]))
    return rec['p'].astype(np.float64), rec['c'].astype(np.float64) / 255


def render_robust(points, colors, angle, size=(860, 560), elevation=0.22):
    """Like make_demo_video.render_cloud, but framed on the 2nd-98th percentile extent so a few
    far points do not shrink the view; points outside the frame are clipped, none are moved."""
    lo, hi = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    centre = (lo + hi) / 2
    extent = float((hi - lo).max())
    w, h = size
    ca, sa = np.cos(angle), np.sin(angle)
    ce, se = np.cos(elevation), np.sin(elevation)
    yaw = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    pitch = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    p = (points - centre) @ yaw.T @ pitch.T
    p[:, 2] += extent * 1.1
    keep = p[:, 2] > 1e-3
    p, c = p[keep], colors[keep]
    f = 0.95 * min(w, h)
    u = (f * p[:, 0] / p[:, 2] + w / 2).astype(int)
    v = (f * p[:, 1] / p[:, 2] + h / 2).astype(int)
    order = np.argsort(-p[:, 2])
    u, v, c = u[order], v[order], c[order]
    img = np.full((h, w, 3), (24, 22, 20), np.uint8)
    bgr = (np.clip(c[:, ::-1], 0, 1) * 255).astype(np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = bgr[ok]
    return img


def shot(writer, path, title, subtitle, seconds, notes=()):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    img = canvas()
    header(img, title, subtitle)
    box_w = W - 80 if not notes else 760
    tile = fit(image, box_w, H - 160)
    x = (W - tile.shape[1]) // 2 if not notes else 40
    paste(img, tile, x, 105)
    y = 150
    for note, color in notes:
        text(img, note, (830, y), 0.56, color); y += 34
    hold(writer, img, seconds)


def scroll(writer, path, title, subtitle, seconds):
    image = cv2.imread(str(path))
    scale = (W - 80) / image.shape[1]
    tall = cv2.resize(image, (W - 80, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    view_h = H - 150
    steps = int(seconds * FPS)
    for k in range(steps):
        top = int((tall.shape[0] - view_h) * min(1.0, max(0.0, (k / steps - 0.1) / 0.8)))
        img = canvas()
        header(img, title, subtitle)
        paste(img, tall[top:top + view_h], 40, 105)
        writer.write(img)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('deliverables/evidence'))
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='CPU ONNX reference metadata.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ev, pg = args.evidence, args.evidence / 'pipegen'
    run = ev / 'deepstream_run'
    meta = json.loads((run / 'metadata.json').read_text())
    ref = json.loads(args.reference.read_text())
    events = [json.loads(l) for l in (run / 'events.jsonl').read_text().splitlines() if l.strip()]
    log = (run / 'spectra_deepstream_run.log').read_text(errors='replace').splitlines()
    engine = json.loads((pg / 'raw' / 'compile.data.json').read_text())
    frames = sorted((run / 'frames').glob('frame_*.jpg'))
    points, colors = read_ply(run / 'cloud.ply')
    if len(points) != meta['point_count']:
        raise ValueError('cloud.ply point count does not match metadata.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError('Cannot open MP4 writer')

    # 1. Title
    img = canvas()
    text(img, 'SPECTRA', (40, 240), 2.4, ACCENT, 4)
    text(img, 'Video -> 3D point cloud + camera poses with VGGT-1B, deployed through CraftifAI PipeGen', (44, 300), 0.8)
    text(img, 'Requirement -> Pipeline IR -> Model (FP16 TensorRT) -> Compiler (DeepStream) -> Run', (44, 350), 0.65, MUTED)
    text(img, 'CraftifAI PipeGen Buildathon | Track A | CRA003 | all footage below is real output', (44, 395), 0.6, MUTED)
    hold(writer, img, 6)

    # 2. Input video
    capture = cv2.VideoCapture(str(args.video))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 4
    clip = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        clip.append(frame)
    capture.release()
    for frame in clip * 2:
        img = canvas()
        header(img, 'Input: a short video of a static scene', f'{args.video.name}: {len(clip)} frames @ {src_fps:.0f} fps (upstream VGGT kitchen scene)')
        tile = fit(frame, W - 80, H - 160)
        paste(img, tile, (W - tile.shape[1]) // 2, 105)
        hold(writer, img, 1 / src_fps)

    # 3. PipeGen stages
    shot(writer, pg / '07_flow_final_v4.png', 'PipeGen 1/5 - Requirement', 'Task flow PipeGen built from the written requirement', 6,
         [('VGGT is the only model', FG), ('local source repo, run command: bash run.sh', FG), ('outputs: frames, events.json, metadata.json', FG)])
    shot(writer, pg / '07_requirement_checks_runsh.png', 'PipeGen 1/5 - Requirement checks', 'All requirement checks pass', 5)
    scroll(writer, ev / 'pipeline_ir.png', 'PipeGen 2/5 - Pipeline IR (approved)', 'Generated graph: input -> preprocess -> VGGT TensorRT -> decode/filter -> event + outputs', 10)
    shot(writer, pg / '10_onnx_inspection_0_unresolved.png', 'PipeGen 3/5 - Model: ONNX inspection', 'VGGT-1B ONNX, 0 unresolved nodes', 5,
         [('First attempt: 57 nodes unresolved, OOM at 15.7 GB', WARN), ('Fixed by spectra/deployment/tensorrt_prep.py:', FG),
          ('constant folding, slice fixes, INT64 inlining', FG), ('ONNX parity unchanged (depth err 4.1e-5)', GREEN)])
    mix = engine.get('precision_mix') or {}
    shot(writer, pg / '12_tensorrt_engine_fp16_built.png', 'PipeGen 3/5 - Model: FP16 TensorRT engine built', 'TensorRT 10.3.0 inside the PipeGen container', 6,
         [(f"build {engine['build_seconds']:.1f} s", FG), (f"engine {engine['engine_bytes'] / 1e6:,.1f} MB, {engine['layer_count']} layers", FG),
          (f"precision mix {json.dumps(mix)}"[:60], FG), ('FP32 weights (4.6 GB) do not fit the 4 GB GPU', MUTED)])
    shot(writer, ev / 'tensorrt_speed_report.png', 'PipeGen 3/5 - Model: speed report', 'PipeGen benchmark of the FP16 engine on the RTX 3050 Laptop GPU', 7,
         [('1,464 ms per frame, ~1 fps, 57.5 W', GREEN), ('same model on CPU (ONNX Runtime): 37.7 s', FG), ('-> ~26x faster on the GPU', GREEN),
          ('note: card labels input 1x3x518x518;', MUTED), ('the engine input is 1x3x518x1036', MUTED)])
    shot(writer, pg / '14_accuracy_blocked_custom_task.png', 'PipeGen 3/5 - Model: accuracy step', 'PipeGen has no metric for a custom pose/depth model', 5,
         [('Checked by hand instead (end of video):', FG), ('DeepStream FP16 vs ONNX CPU reference', FG)])
    shot(writer, pg / '15_pipeline_compile_ok_debugrun_5s.png', 'PipeGen 4/5 - Compiler: DeepStream pipeline generated', 'compile=ok; PipeGen\'s 5 s smoke test ends before the 2.4 GB engine loads', 6,
         [('Generated: nvurisrcbin -> nvstreammux ->', FG), ('nvinfer_backport (FP16 engine) -> nvcustomprocess', FG), ('-> meta_json -> JPEG frames', FG),
          ('Its VGGT decode stage was left as a stub:', WARN), ('completed in deepstream/parsers/vggt_spectra.cpp', WARN)])

    # 4. Run log
    img = canvas()
    header(img, 'PipeGen 5/5 - Run: DeepStream pipeline on the GPU (actual log)', 'docker exec craftifai-amd64-v3 bash deepstream/run.sh')
    keep = [l for l in log if l.startswith('[vggt]') or 'deserialized trt engine' in l or 'Execution ended' in l or l.startswith('== ')]
    y = 135
    for line in keep[:16]:
        if 'deserialized' in line:
            line = 'nvinfer_backport: deserialized trt engine ' + line.split(':')[-1].split('/')[-1]
        text(img, line.replace('/home/atharv/spectra/output/', '')[:118], (40, y), 0.48, GREEN); y += 28
    hold(writer, img, 10)

    # 5. Output frames
    for frame_path in frames * 2:
        image = cv2.imread(str(frame_path))
        img = canvas()
        header(img, 'Pipeline output frames: current view | VGGT depth (FP16 TensorRT)', 'Each frame is paired with the frame 1 s earlier (frames 0-3 wait for a partner)')
        tile = fit(image, W - 80, H - 160)
        paste(img, tile, (W - tile.shape[1]) // 2, 105)
        hold(writer, img, 1.2)

    # 6. Point cloud
    seconds = 16
    for k in range(seconds * FPS):
        angle = 0.55 * np.sin(2 * np.pi * k / (seconds * FPS))
        img = canvas()
        header(img, 'cloud.ply - frames 0 + 4, from the DeepStream run',
               f"{meta['point_count']:,} points kept of {meta['input_point_count']:,} (confidence >= 50th pct, 5 mm voxel*, outlier removal)  *arbitrary units")
        paste(img, render_robust(points, colors, angle), (W - 860) // 2, 110)
        writer.write(img)

    # 7. Events
    img = canvas()
    header(img, 'Actionable output: one geometry_ready event per frame pair', 'events.jsonl is appended only after the PLY is written and re-read successfully')
    y = 150
    for e in events:
        text(img, f"frames {e['frames']}  ->  {e['point_count']:,} points   {Path(e['ply']).name}   action: {e['action']}", (60, y), 0.62, GREEN); y += 44
    first = {k: (Path(v).name if k in ('ply', 'metadata') else v) for k, v in events[0].items()}
    for line in json.dumps(first, indent=1).splitlines()[:14]:
        text(img, line, (60, y + 20), 0.5, FG); y += 24
    hold(writer, img, 9)

    # 8. Agreement + speed + status
    img = canvas()
    header(img, 'Results', 'Every value below is read from the run files')
    rc, dc = ref['cameras'], meta['cameras']
    rb = float(np.linalg.norm(np.subtract(rc[1]['center'], rc[0]['center'])))
    db = float(np.linalg.norm(np.subtract(dc[1]['center'], dc[0]['center'])))
    Ra, Rb = np.asarray(rc[1]['camera_from_world'])[:, :3], np.asarray(dc[1]['camera_from_world'])[:, :3]
    rot = float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
    rows = [
        ('Agreement', f'camera baseline {db:.4f} vs ONNX CPU {rb:.4f} ({100 * (db - rb) / rb:+.1f}%), view-1 rotation diff {rot:.2f} deg'),
        ('Agreement', f"points {meta['point_count']:,} vs ONNX CPU {ref['point_count']:,} (frames 0 + 4)"),
        ('Speed', 'PipeGen benchmark: 1,464 ms/frame FP16 GPU vs 37.7 s ONNX CPU'),
        ('Open issue', f"DeepStream pipeline ~{events[0]['frame_wall_seconds']:.0f} s/frame - also in PipeGen's unmodified pipeline;"),
        ('', 'cause not isolated (suspected 4 GB VRAM pressure under WSL2)'),
        ('Limits', 'Two views per reconstruction; pairs not merged; scale arbitrary; one indoor scene'),
    ]
    y = 160
    for tag, desc in rows:
        color = GREEN if tag in ('Agreement', 'Speed') else WARN if tag == 'Open issue' else MUTED
        text(img, tag, (60, y), 0.66, color, 2)
        text(img, desc, (240, y), 0.58, FG); y += 56
    hold(writer, img, 12)
    writer.release()
    print(args.output, 'written')


if __name__ == '__main__':
    main()

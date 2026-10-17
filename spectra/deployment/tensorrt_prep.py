"""Make a validated VGGT bundle parseable by TensorRT without changing its math.

PipeGen's ONNX inspection (TensorRT parser) left 57 nodes unresolved in the exported graph:
- 50 Einsum/Sin/Cos/Concat/Cast nodes of upstream make_sincos_pos_embed in the DPT head and
  4 Slices of the learned camera/register tokens. None depend on the `images` input, so their
  values are computed once here (ONNX Runtime) and stored as initializers.
- Slices whose `ends` is INT64_MAX ("to the end"). With the fixed 518x1036 input every sliced
  axis has a static size, so INT64_MAX is replaced by that size.
The result is re-run in ONNX Runtime and compared with the original PyTorch reference outputs
at the original tolerances before a manifest is written.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .contract import INPUT_NAME, OUTPUT_NAMES, OUTPUT_SHAPES, sha256

INT64_MAX = np.iinfo(np.int64).max
FOLD_OPS = {'Einsum', 'Sin', 'Cos', 'Concat', 'Cast', 'Slice'}


def input_dependent(graph):
    dynamic = {i.name for i in graph.input}
    for node in graph.node:
        if node.op_type != 'Constant' and any(i in dynamic for i in node.input if i):
            dynamic.update(node.output)
    return dynamic


def fold_targets(graph, dynamic):
    """Outputs of constant FOLD_OPS nodes that feed input-dependent nodes."""
    folded = {o for n in graph.node if n.op_type in FOLD_OPS and n.output[0] not in dynamic for o in n.output}
    targets = []
    for node in graph.node:
        if node.op_type in FOLD_OPS and node.output[0] not in dynamic:
            continue
        targets += [i for i in node.input if i in folded and i not in targets]
    targets += [o.name for o in graph.output if o.name in folded and o.name not in targets]
    return targets


def evaluate_constants(graph, targets, base_dir, opset, ir_version):
    """Run only the constant ancestors of `targets` in ONNX Runtime."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper
    producer = {o: n for n in graph.node for o in n.output}
    inits = {t.name: t for t in graph.initializer}
    needed_nodes, needed_inits, stack, seen = [], [], list(targets), set()
    while stack:
        name = stack.pop()
        if name in seen or not name:
            continue
        seen.add(name)
        if name in inits:
            needed_inits.append(name)
        elif name in producer:
            node = producer[name]
            if node not in needed_nodes:
                needed_nodes.append(node)
                stack.extend(node.input)
    order = {id(n): i for i, n in enumerate(graph.node)}
    needed_nodes.sort(key=lambda n: order[id(n)])
    tensors = [numpy_helper.from_array(numpy_helper.to_array(inits[n], base_dir=str(base_dir)), n) for n in needed_inits]
    outputs = [helper.make_empty_tensor_value_info(t) for t in targets]
    sub = helper.make_graph(needed_nodes, 'vggt_constants', [], outputs, tensors)
    model = helper.make_model(sub, opset_imports=[helper.make_opsetid('', opset)], ir_version=ir_version)
    session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    return dict(zip(targets, session.run(targets, {}))), len(needed_nodes)


def static_shapes(model):
    import onnx
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = v.type.tensor_type.shape.dim
        if dims and all(d.HasField('dim_value') for d in dims):
            shapes[v.name] = [d.dim_value for d in dims]
    for t in inferred.graph.initializer:
        shapes[t.name] = list(t.dims)
    return shapes


def clamp_slice_ends(model, shapes):
    """Replace INT64_MAX slice ends by the static axis size (identical result)."""
    from onnx import numpy_helper
    graph = model.graph
    small = {t.name: numpy_helper.to_array(t) for t in graph.initializer
             if t.data_type == 7 and not t.data_location and np.prod(t.dims) <= 8}
    added, changed = {}, []
    for node in graph.node:
        if node.op_type != 'Slice' or len(node.input) < 3 or node.input[2] not in small:
            continue
        ends = small[node.input[2]]
        if not (ends == INT64_MAX).any():
            continue
        data = shapes.get(node.input[0])
        if data is None:
            raise ValueError(f'{node.name}: INT64_MAX end on a tensor without a static shape')
        axes = small[node.input[3]] if len(node.input) > 3 and node.input[3] else np.arange(len(ends))
        new = np.array([data[int(a)] if e == INT64_MAX else e for e, a in zip(ends, axes)], dtype=np.int64)
        key = f'trt_slice_end_{"_".join(map(str, new.tolist()))}'
        if key not in added:
            added[key] = numpy_helper.from_array(new, key)
            graph.initializer.append(added[key])
        node.input[2] = key
        changed.append(node.name)
    return changed


def unshare_slice_parameters(model):
    """Give each Slice its own starts/ends/axes/steps initializers.

    The exporter reuses one tensor (e.g. val_3 = [1]) as ends, axes and steps of the same node;
    PipeGen's per-node TensorRT probe rejects such nodes. Copies have identical values.
    """
    from onnx import numpy_helper
    graph = model.graph
    inits = {t.name: t for t in graph.initializer if not t.data_location}
    roles = ('starts', 'ends', 'axes', 'steps')
    changed = []
    for node in graph.node:
        if node.op_type != 'Slice':
            continue
        names = [i for i in node.input[1:] if i]
        if len(names) == len(set(names)):
            continue
        for position, role in enumerate(roles, start=1):
            if position >= len(node.input) or node.input[position] not in inits:
                continue
            copy = numpy_helper.from_array(numpy_helper.to_array(inits[node.input[position]]), f'{node.name}_{role}')
            graph.initializer.append(copy)
            node.input[position] = copy.name
        changed.append(node.name)
    return changed


def inline_int64_initializers(path):
    """Keep INT64 initializers inside the .onnx; PipeGen's probe cannot import them externally."""
    import onnx
    from onnx.external_data_helper import load_external_data_for_tensor
    model = onnx.load(str(path), load_external_data=False)
    moved = []
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.INT64 and tensor.data_location == onnx.TensorProto.EXTERNAL:
            load_external_data_for_tensor(tensor, str(path.parent))
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
            moved.append(tensor.name)
    if moved:
        onnx.save_model(model, str(path))
    return moved


def prepare(source, destination, rtol, atol):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Destination must be empty; use a new directory to preserve prior evidence')
    original_manifest = json.loads((source / 'model.json').read_text())
    if original_manifest['files']['vggt.onnx'] != sha256(source / 'vggt.onnx'):
        raise ValueError('Source vggt.onnx does not match its manifest')
    start = time.perf_counter()
    model = onnx.load(str(source / 'vggt.onnx'), load_external_data=False)
    graph = model.graph
    opset = next(o.version for o in model.opset_import if o.domain in ('', 'ai.onnx'))
    dynamic = input_dependent(graph)
    targets = fold_targets(graph, dynamic)
    values, evaluated = evaluate_constants(graph, targets, source, opset, model.ir_version)
    for name, value in values.items():
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
    producers = {o: n for n in graph.node for o in n.output}
    for node in {id(producers[t]): producers[t] for t in targets}.values():
        graph.node.remove(node)
    # Dead-code elimination for nodes/initializers no longer reachable from the outputs.
    live = {o.name for o in graph.output}
    keep = []
    for node in reversed(list(graph.node)):
        if any(o in live for o in node.output):
            keep.append(node)
            live.update(i for i in node.input if i)
    removed_nodes = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(reversed(keep))
    unused = [t for t in graph.initializer if t.name not in live]
    for t in unused:
        graph.initializer.remove(t)
    del graph.value_info[:]
    changed = clamp_slice_ends(model, static_shapes(model))
    unshared = unshare_slice_parameters(model)
    # Restore per-tensor shape/dtype annotations; PipeGen's per-node TensorRT check needs them.
    model = onnx.shape_inference.infer_shapes(model)
    missing = [v.name for v in model.graph.value_info if not v.type.tensor_type.HasField('shape')]
    if len(model.graph.value_info) == 0 or missing:
        raise ValueError(f'Shape inference left {len(missing)} tensors without shapes')
    # Resolve external weights from the source bundle, then write a fresh external-data pair.
    onnx.load_external_data_for_model(model, str(source))
    path = destination / 'vggt.onnx'
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location='vggt.onnx.data', size_threshold=1024)
    operators = sorted({n.op_type for n in model.graph.node})
    remaining = sorted({n.name for n in model.graph.node if n.op_type in {'Einsum', 'Sin', 'Cos'}})
    del model, graph
    inlined = inline_int64_initializers(path)
    onnx.checker.check_model(str(path))
    for name in ('reference-input.npy', 'reference-output.npz'):
        (destination / name).write_bytes((source / name).read_bytes())
    atlas = np.load(destination / 'reference-input.npy')
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(OUTPUT_NAMES, {INPUT_NAME: atlas})
    errors = {}
    with np.load(destination / 'reference-output.npz') as reference:
        for name, value in zip(OUTPUT_NAMES, actual):
            if value.shape != OUTPUT_SHAPES[name]:
                raise ValueError(f'Unexpected output shape: {name} {value.shape}')
            np.testing.assert_allclose(value, reference[name], rtol=rtol, atol=atol)
            errors[name] = {'max_absolute_error': float(np.max(np.abs(value - reference[name])))}
    files = {p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'model.json'}
    manifest = dict(original_manifest)
    manifest.update({
        'operators': operators,
        'files': files,
        'tensorrt_preparation': {
            'source_bundle_vggt_onnx_sha256': original_manifest['files']['vggt.onnx'],
            'source_bundle_data_sha256': original_manifest['files']['vggt.onnx.data'],
            'constant_tensors_folded': sorted(values),
            'constant_nodes_evaluated': evaluated,
            'nodes_removed': removed_nodes,
            'initializers_removed': len(unused),
            'slice_ends_clamped': changed,
            'slice_parameters_unshared': unshared,
            'int64_initializers_inlined': inlined,
            'remaining_einsum_sin_cos_nodes': remaining,
        },
        'validation': {'onnx_reference_passed': True, 'specialization_parity_passed':
                       original_manifest['validation']['specialization_parity_passed'],
                       'compared_against': 'original PyTorch reference-output.npz',
                       'rtol': rtol, 'atol': atol, 'errors': errors,
                       'seconds': time.perf_counter() - start},
        'pipegen_verified': False, 'tensorrt_verified': False,
    })
    (destination / 'model.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Validated bundle (vggt.onnx, vggt.onnx.data, model.json, reference-*)')
    parser.add_argument('output', type=Path)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.rtol, args.atol)
    print(json.dumps({k: manifest[k] for k in ('tensorrt_preparation', 'validation', 'files')}, indent=2))


if __name__ == '__main__':
    main()

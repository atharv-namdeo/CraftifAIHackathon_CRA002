"""Does TensorRT resolve vggt.onnx.data when parsing from bytes, from a path, or from another cwd?"""
import os, sys
import tensorrt as trt

path = sys.argv[1]
logger = trt.Logger(trt.Logger.ERROR)

def attempt(label, fn):
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    ok = fn(parser)
    errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
    print(f'{label}: ok={ok} layers={network.num_layers} errors={errs[:1]}', flush=True)

data = open(path, 'rb').read()
os.chdir('/opt/PipeGen')
attempt('parse(bytes) cwd=/opt/PipeGen', lambda p: p.parse(data))
attempt('parse(bytes, path) cwd=/opt/PipeGen', lambda p: p.parse(data, path))
os.chdir(os.path.dirname(path))
attempt('parse(bytes) cwd=model dir', lambda p: p.parse(data))

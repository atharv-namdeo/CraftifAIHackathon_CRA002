"""Parse an ONNX file with the TensorRT in the PipeGen container; report every parser error."""
import sys, time
import tensorrt as trt

path = sys.argv[1]
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
start = time.time()
ok = parser.parse_from_file(path)
print(f'tensorrt {trt.__version__} parse_from_file({path}) -> {ok} in {time.time() - start:.1f}s')
print('errors:', parser.num_errors)
for i in range(min(parser.num_errors, 20)):
    e = parser.get_error(i)
    print(f'  [{i}] code={e.code()} node={e.node()} name={e.node_name()} op={e.node_operator()} desc={e.desc()}')
print('network layers:', network.num_layers, 'inputs:', network.num_inputs, 'outputs:', network.num_outputs)

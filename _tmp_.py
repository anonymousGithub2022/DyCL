import os

import torchvision
import torch
import onnx
import tvm
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

import tmp.trt_common as trt_common




from tvm import relay
from tvm.contrib import graph_executor


TRT_LOGGER = trt.Logger()

print(os.getcwd())
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

print(tvm.__version__)
model = torchvision.models.resnet50(pretrained=False)

device_id = 0


input_shape = [1, 3, 224, 224]
onnx_path = '__test__.onnx'
input_data = torch.randn(input_shape)
torch.onnx.export(
    model,
    input_data,
    onnx_path,       # where to save the model
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,    # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['x'],
    output_names=['y']
)
onnx_model = onnx.load_model(onnx_path)

scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "x"
shape_list = [(input_name, input_data.shape)]
mod, params = relay.frontend.from_onnx(onnx_model)

if device_id == 0:
    target = tvm.target.cuda('P6000 -libs=cudnn -arch=sm_61')
    dev = tvm.device('cuda')
    device = 'cuda'
else:
    target = tvm.target.Target('llvm')
    dev = tvm.device('cpu')
    device = 'cpu'


with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs

model = model.eval().to(device)
input_data = input_data.to(device)
t1 = time.time()
for _ in range(100):
    torch_y = model(input_data)
t2 = time.time()
torch_time = t2 - t1


tvm_input = tvm.nd.array(
    input_data.detach().cpu().numpy(), device=dev
)
t1 = time.time()
for _ in range(100):
    m.set_input(input_name, tvm_input)
    m.run()
    tvm_output = m.get_output(0)
t2 = time.time()
tvm_time = t2 - t1

scripted_model = scripted_model.eval().to(device)
t1 = time.time()
for _ in range(100):
    _ = scripted_model(input_data)
t2 = time.time()
script_time = t2 - t1


t1 = time.time()
for _ in range(100):
    _ = scripted_model(input_data)
t2 = time.time()
script_time = t2 - t1

onnx_device = ['CUDAExecutionProvider']
t1 = time.time()
onnx_session = onnxruntime.InferenceSession(onnx_path, providers=onnx_device)
input_feed = {'x': input_data.detach().cpu().numpy()}
for _ in range(100):
    o = onnx_session.run(['y'], input_feed=input_feed)
print(o)
t2 = time.time()
onnx_time = t2 - t1


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                trt_common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    else:
        return build_engine()


with get_engine(onnx_path, './tensorrt.pb') as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = trt_common.allocate_buffers(engine)
    t1 = time.time()
    for _ in range(100):
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
    t2 = time.time()

    stream.synchronize()
trt_time_v1 = t2 - t1


engine = get_engine(onnx_path, './tensorrt.pb')
context = engine.create_execution_context()
inputs, outputs, bindings, stream = trt_common.allocate_buffers(engine)
# [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

t1 = time.time()
for _ in range(100):
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
t2 = time.time()
# [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
# Synchronize the stream
stream.synchronize()


trt_time_v2 = t2 - t1

print(tvm_time, torch_time, script_time, onnx_time, trt_time_v1, trt_time_v2)







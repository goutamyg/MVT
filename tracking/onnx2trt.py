# ref: https://github.com/nikil-ravi/trt_tutorial/blob/master/braggnn-pytorch/build_engine.py
import engine as eng
from onnx import ModelProto
import tensorrt as trt
import os

def load_engine(trt_runtime, plan_path):
    trt.init_libnvinfer_plugins(None, "")
    with open(plan_path, 'rb') as f:
       engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


tracker_name = 'MobileViT_Track_ep0100'
tracker_config = 'mobilevit_256_64x2_got10k_ep100_cosine_annealing'

onnx_path = os.path.join('output/checkpoints/train/mobilevit_track', tracker_config, '{}.onnx'.format(tracker_name))
engine_name = os.path.join('output/checkpoints/train/mobilevit_track', tracker_config, '{}_FP16_TRT.plan'.format(tracker_name))
batch_size = 1

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0_0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d0_1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d0_2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

d1_0 = model.graph.input[1].type.tensor_type.shape.dim[1].dim_value
d1_1 = model.graph.input[1].type.tensor_type.shape.dim[2].dim_value
d1_2 = model.graph.input[1].type.tensor_type.shape.dim[3].dim_value

shape = [[batch_size, d0_0, d0_1, d0_2], [batch_size, d1_0, d1_1, d1_2]]
engine = eng.build_engine(onnx_path, shape=shape)
eng.save_engine(engine, engine_name)
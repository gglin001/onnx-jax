import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime

from onnx_jax.backend import run_model

from tests.tools import _cosin_sim, _gen_random

# https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
model_fp = '/home/allen/ml_data/models/onnx/efficientnet-lite4-11.onnx'

# load model
onnx_model = onnx.load_model(model_fp)
graph = onnx_model.graph

# input
input_shape = [x.dim_value for x in graph.input[0].type.tensor_type.shape.dim]
input_ = _gen_random(shape=input_shape)
input_dict = {graph.input[0].name: input_}

# onnx-jax
outputs_jax = run_model(onnx_model, input_dict)
outputs_jax = outputs_jax[0].flatten()
print(f"outputs_jax.shape: {outputs_jax.shape}")
top_5_idx = outputs_jax.argsort()[:5]
print(f"onnx-jax result: \n\ttop_5_idx: {top_5_idx}, top_5_score: {outputs_jax[top_5_idx]}")

# onnxruntime
sess = onnxruntime.InferenceSession(model_fp)
outputs_ort = sess.run(None, {graph.input[0].name: np.asarray(input_)})
outputs_ort = outputs_ort[0].flatten()
print(f"outputs_ort.shape: {outputs_ort.shape}")
top_5_idx = outputs_ort.argsort()[:5]
print(f"onnxruntime result: \n\ttop_5_idx: {top_5_idx}, top_5_score: {outputs_ort[top_5_idx]}")

# sim
sim = _cosin_sim(jnp.asarray(outputs_ort[0]), outputs_jax[0])
print(f"sim: {sim}")

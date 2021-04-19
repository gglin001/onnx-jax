import cv2
import jax.numpy as jnp
import numpy as np
import onnx

from onnx_jax.backend import run_model
from onnx_jax.logger import logger


def _cosin_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    cos_sim = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return cos_sim


# https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
fp = '/home/allen/ml_data/models/onnx/mobilenetv2-7.onnx'
onnx_model = onnx.load_model(fp)
graph = onnx_model.graph

# input_shape is [1,3,224,244]
input_shape = [x.dim_value for x in graph.input[0].type.tensor_type.shape.dim]
logger.info(f"input shape: {input_shape}")

# https://commons.wikimedia.org/wiki/File:Giant_Panda_in_Beijing_Zoo_1.JPG
# imagenet id of panda is 388
fp_img = '/home/allen/ml_data/data/Giant_Panda_in_Beijing_Zoo_1.jpeg'
input_ = cv2.imread(fp_img, 1)
input_ = cv2.resize(input_, (224, 224))
input_ = jnp.expand_dims(input_, 0)
input_ = jnp.transpose(input_, [0, 3, 1, 2])
input_ = input_.astype(jnp.float32)
input_ = input_ / 255.0
input_ = input_ - jnp.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
input_ = input_ / jnp.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])

# run model
outputs = run_model(onnx_model, [input_])
logger.info(f"onnx-jax result: {jnp.argmax(outputs[0])}")

try:
    import onnxruntime as ort

    sess = ort.InferenceSession(fp)
    sess.get_inputs()
    out = sess.run(None, {graph.input[0].name: np.asarray(input_)})
    logger.info(f"onnxruntime reult: {np.argmax(out[0])}")

    # compare with onnxruntime reult
    sim = _cosin_sim(jnp.asarray(out[0]), outputs[0])
    logger.info(f"Output tensor similarity with onnxruntime: {sim}")
except:
    pass

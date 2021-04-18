# onnx-jax

[onnx](https://github.com/onnx/onnx) runtime based on [jax](https://github.com/google/jax)

[comment]: <> (This project is deeply influenced by much Great Software:)

## Quick start

```
import onnx
from onnx_jax.backend import run_model

onnx_model = onnx.load_model('/path/to/model.onnx')
outputs = run_model(onnx_model, [input_in_numpy])

```

## Demo

[demo.ipynb](tests/demo.ipynb)

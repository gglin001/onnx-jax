from typing import Sequence

import jax.numpy as jnp
from onnx import defs, numpy_helper
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
from onnx.helper import make_opsetid

from onnx_jax.handler_helper import get_all_backend_handlers
from onnx_jax.pb_wrapper import OnnxNode, build_ref_dict


class JaxRep(BackendRep):
    def __init__(self, model=None):
        super(JaxRep, self).__init__()

        self.model = model

    def run(self, inputs, **kwargs):
        return JaxBackend.run_model(self.model, inputs)


class JaxBackend(Backend):
    @classmethod
    def supports_device(cls, device):
        if device == 'CPU':
            return True

        return False

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):

        return JaxRep(model)

    @classmethod
    def run_node(cls, node, inputs, device='CPU', **kwargs):
        onnx_node = OnnxNode(node)
        jit_func = cls._jit(onnx_node, **kwargs)
        inputs = [jnp.asarray(x) for x in inputs]
        # TODO support uncertain number inputs, like concat
        outputs = jit_func(*inputs, *onnx_node.attrs_list)
        return outputs if isinstance(outputs, Sequence) else [outputs]

    @classmethod
    def run_model(cls, model, inputs, device='CPU', **kwargs):
        def _asarray(proto):
            return jnp.asarray(numpy_helper.to_array(proto).reshape(tuple(proto.dims)))

        tensor_ref_dict = build_ref_dict(model)
        graph = model.graph
        if model.ir_version < 3:
            opset = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset = model.opset_import

        if isinstance(inputs, dict):
            tensor_dict = dict(
                {k: v for k, v in inputs.items()},
                **{n.name: _asarray(n) for n in graph.initializer},
            )
        else:
            graph_inputs = [x.name for x in graph.input]
            tensor_dict = dict(
                {k: v for k, v in zip(graph_inputs, inputs)},
                **{n.name: _asarray(n) for n in graph.initializer},
            )

        jit_funcs = {}
        onnx_nodes = {}
        for node in graph.node:
            onnx_node = OnnxNode(node)
            jit_func = cls._jit(onnx_node, **kwargs)
            jit_funcs[node.name] = jit_func
            onnx_nodes[node.name] = onnx_node

        ref_dict = {}
        for node in graph.node:
            onnx_node = onnx_nodes[node.name]
            print(f"running: {node.op_type}, {node.name}")

            node_inputs = [tensor_dict[x] for x in node.input]
            jit_func = jit_funcs[node.name]
            outputs = jit_func(*node_inputs, *onnx_node.attrs_list)
            outputs = outputs if isinstance(outputs, Sequence) else [outputs]

            for name, output in zip(node.output, outputs):
                tensor_dict[name] = output

                node_input_shapes = [tensor_dict[x].shape for x in node.input]
                node_output_shapes = [tensor_dict[x].shape for x in node.output]
                print(f"\t{node_input_shapes} -> {node_output_shapes}")

                for input_ in node.input:
                    if input_ in ref_dict:
                        ref_dict[input_] += 1
                    else:
                        ref_dict[input_] = 1
                remove_keys = []
                for k, v in ref_dict.items():
                    if tensor_ref_dict[k] == v:
                        remove_keys.append(k)
                for rm_k in remove_keys:
                    del ref_dict[rm_k]
                    del tensor_dict[rm_k]

        return [tensor_dict[n.name] for n in graph.output]

    @classmethod
    def _jit(cls, node, opset=None, handlers=None, **kwargs):
        handlers = handlers or cls._get_handlers(opset)
        if handlers:
            handler = (
                handlers[node.domain].get(node.op_type, None)
                if node.domain in handlers
                else None
            )
            if handler:
                return handler.handle(node, inputs=None, **kwargs)

        raise BackendIsNotSupposedToImplementIt(
            "{} is not implemented.".format(node.op_type)
        )

    @classmethod
    def _get_handlers(cls, opset):
        opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        opset_dict = dict([(o.domain, o.version) for o in opset])
        return get_all_backend_handlers(opset_dict)


prepare = JaxBackend.prepare

run_node = JaxBackend.run_node

run_model = JaxBackend.run_model

supports_device = JaxBackend.supports_device

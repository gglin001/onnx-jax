from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
from onnx.helper import make_opsetid

from onnx_jax.common.handler_helper import get_all_backend_handlers
from onnx_jax.pb_wrapper import OnnxNode


class JaxBackend(Backend):

    @classmethod
    def prepare(cls, model, **kwargs):
        super(JaxBackend, cls).prepare(model, **kwargs)

        pass

    @classmethod
    def run_node(cls, node, inputs, **kwargs):
        super(JaxBackend, cls).run_node(node, inputs)

        outputs = cls._run_node_imp(OnnxNode(node), inputs, **kwargs)
        return outputs

    @classmethod
    def run_model(cls, model, input_dict, **kwargs):
        def _asarray(proto):
            return numpy_helper.to_array(proto).reshape(tuple(proto.dims))

        graph = model.graph
        if model.ir_version < 3:
            opset = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset = model.opset_import

        tensor_dict = dict({k: v for k, v in input_dict.items()},
                           **{n.name: _asarray(n) for n in graph.initializer})
        handlers = cls._get_handlers(opset)
        for node in graph.node:
            node_inputs = [tensor_dict[x] for x in node.input]

            print(f"running: {node.op_type}, {node.name}")
            outputs = cls._run_node_imp(OnnxNode(node), node_inputs, opset, handlers)
            for name, output in zip(node.output, outputs):
                tensor_dict[name] = output

        return [tensor_dict[n.name] for n in graph.output]

    @classmethod
    def _run_node_imp(cls, node, inputs, opset=None, handlers=None, **kwargs):
        handlers = handlers or cls._get_handlers(opset)
        if handlers:
            handler = handlers[node.domain].get(node.op_type, None) if node.domain in handlers else None
            if handler:
                return handler.handle(node, inputs=inputs, **kwargs)

        raise BackendIsNotSupposedToImplementIt("{} is not implemented.".format(node.op_type))

    @classmethod
    def _get_handlers(cls, opset):
        opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        opset_dict = dict([(o.domain, o.version) for o in opset])
        return get_all_backend_handlers(opset_dict)


prepare = JaxBackend.prepare

run_node = JaxBackend.run_node

run_model = JaxBackend.run_model

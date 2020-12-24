from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
from onnx.helper import make_opsetid

from onnx_jax.common.handler_helper import get_all_backend_handlers
from onnx_jax.pb_wrapper import OnnxNode


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
        super(JaxBackend, cls).run_node(node, inputs)

        outputs = cls._run_node_imp(OnnxNode(node), inputs, **kwargs)
        return outputs

    @classmethod
    def run_model(cls, model, inputs, device='CPU', **kwargs):
        def _asarray(proto):
            return numpy_helper.to_array(proto).reshape(tuple(proto.dims))

        graph = model.graph
        if model.ir_version < 3:
            opset = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset = model.opset_import

        if isinstance(inputs, dict):
            tensor_dict = dict({k: v for k, v in inputs.items()},
                               **{n.name: _asarray(n) for n in graph.initializer})
        else:
            graph_inputs = [x.name for x in graph.input]
            tensor_dict = dict({k: v for k, v in zip(graph_inputs, inputs)},
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

supports_device = JaxBackend.supports_device

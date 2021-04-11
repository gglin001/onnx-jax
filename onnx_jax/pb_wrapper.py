def convert_onnx(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.HasField('g'):
        return attr_proto.g
    elif attr_proto.floats:
        return tuple(attr_proto.floats)
    elif attr_proto.ints:
        return tuple(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        str_list = tuple(map(lambda x: str(x, 'utf-8'), str_list))
        return str_list
    elif attr_proto.HasField('sparse_tensor'):
        return attr_proto.sparse_tensor
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name, convert_onnx(attr)) for attr in node.attribute])
        self.attrs_list = []
        self.inputs = list(node.input)
        self.len_inputs = len(node.input)
        self.outputs = list(node.output)
        self.len_outputs = len(node.output)
        self.node_proto = node


def build_ref_dict(model):
    ref_dict = {}
    for node in model.graph.node:
        inputs = node.input
        for input_ in inputs:
            if input_ in ref_dict:
                ref_dict[input_] += 1
            else:
                ref_dict[input_] = 1

    return ref_dict

import inspect
import logging

from onnx import defs
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt


class Handler(object):
    """ This class is base handler class.
    Base backend and frontend base handler class inherit this class.

    All operator handler MUST put decorator @onnx_op to register corresponding op.
    """

    ONNX_OP = None

    DOMAIN = defs.ONNX_DOMAIN
    VERSION = 0
    SINCE_VERSION = 0

    @classmethod
    def check_cls(cls):
        if not cls.ONNX_OP:
            logging.warning(
                "{} doesn't have ONNX_OP. "
                "Please use Handler.onnx_op decorator to register ONNX_OP.".format(
                    cls.__name__))

    @classmethod
    def args_check(cls, node, **kwargs):
        """ Check args. e.g. if shape info is in graph.
        Raise exception if failed.

        :param node: NodeProto for backend.
        :param kwargs: Other args.
        """
        pass

    @classmethod
    def handle(cls, node, **kwargs):
        """ Main method in handler. It will find corresponding versioned handle method,
        whose name format is `version_%d`. So prefix `version_` is reserved in onnx-tensorflow.
        DON'T use it for other purpose.

        :param node: NodeProto for backend.
        :param kwargs: Other args.
        :return: TensorflowNode for backend.
        """
        ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
        if ver_handle:
            cls.args_check(node, **kwargs)
            return ver_handle(node, **kwargs)

        raise BackendIsNotSupposedToImplementIt(
            "{} version {} is not implemented.".format(node.op_type, cls.SINCE_VERSION))

    @classmethod
    def get_versions(cls):
        """ Get all support versions.

        :return: Version list.
        """
        versions = []
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("version_"):
                versions.append(int(k.replace("version_", "")))
        return versions

    @staticmethod
    def onnx_op(op):
        return Handler.property_register("ONNX_OP", op)

    @staticmethod
    def domain(d):
        return Handler.property_register("DOMAIN", d)

    @staticmethod
    def property_register(name, value):

        def deco(cls):
            setattr(cls, name, value)
            return cls

        return deco


domain = Handler.domain
onnx_op = Handler.onnx_op
property_register = Handler.property_register

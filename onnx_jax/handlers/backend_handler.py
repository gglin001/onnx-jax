from .handler import Handler


class BackendHandler(Handler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {}

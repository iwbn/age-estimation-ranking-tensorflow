import abc
import tensorflow as tf


class Pretrained:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        assert type(name) == str
        self.name = str(name)

    @abc.abstractmethod
    def construct(self, *inputs, **options):
        pass

    @abc.abstractmethod
    def load_pretrained(self, *inputs, **options):
        pass

    @abc.abstractproperty
    def input_shape(self):
        pass

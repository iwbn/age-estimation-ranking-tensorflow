import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os.path
from .FaceNet import inception_resnet_v1

from .pretrained_base import Pretrained


class FaceNet(Pretrained):
    class OPTS:
        def __init__(self):
            self.image_mean = np.reshape([117.0, 117.0, 117.0], [1, 1, 1, 3])#np.reshape([127.5, 127.5, 127.5], [1, 1, 1, 3])
            self.weight_path = None
            self.weight_decay = 1e-4

    def __init__(self, opts=None):
        Pretrained.__init__(self, "FaceNet")
        if opts is None:
            self.opts = self.OPTS()
        else:
            self.opts = opts

    @property
    def input_shape(self):
        return (None, 160, 160, 3)

    def normalize(self, x):
        res = x - self.opts.image_mean
        return res

    def construct(self, x, is_training, keep_prob=1., opts=None):
        if opts is None:
            opts = self.opts

        with tf.variable_scope(self.name):
            with tf.control_dependencies([tf.assert_type(x, tf.float32),
                                            tf.assert_equal(tf.shape(x)[1::], self.input_shape[1::])]):

                # self.x = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')

                self.pre_logits, self.end_points = inception_resnet_v1.inference(self.normalize(x), keep_prob,
                                                                                 phase_train=is_training)

                self.bottleneck = self.pre_logits
                return self.bottleneck

    def load_pretrained(self, session, scopes, weight_path, regression=False):
        data_dict = np.load(weight_path, encoding='latin1').item()
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        var_idx = {vars[i].name: i for i in range(len(vars))}
        ops = []
        for key in data_dict.keys():
            for scope in scopes:
                if key.startswith(scope):
                    data = data_dict[key]
                    scope_key = key
                    try:
                        var = vars[var_idx["%s/%s" %(self.name, scope_key)]]
                    except KeyError:
                        print ("%s unused (no variable)" % (scope_key))
                        continue
                    try:
                        if len(var.get_shape()) == 5:
                            data = np.expand_dims(data, 0)
                        elif len(var.get_shape()) == 4 and var.get_shape().as_list()[2] != data.shape[2]:
                            data = np.concatenate((data, data), axis=2) / 2.0
                        if regression and len(var.get_shape()) == 2 and var.get_shape().as_list()[1] != data.shape[
                            1]:
                            data = data[:, 0:var.get_shape().as_list()[1]]
                        ops.append(var.assign(data))

                        print ("%s used" % (scope_key))
                    except ValueError:
                        print ("%s unused" % (scope_key))
                        pass
        session.run(ops)

class FaceNetLarge(FaceNet):
    @property
    def input_shape(self):
        return (None, 299, 299, 3)


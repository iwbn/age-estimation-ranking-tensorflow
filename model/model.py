import tensorflow as tf
import abc


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper

class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        self.__opt_assign_ops = {}
        self.learning_rate = self.option_variable(1e-4, tf.float32, 'learning_rate')
        self.global_step = self.option_variable(0, tf.int32, 'global_step')
        tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step)
        self.set_additional_options()
        self.build()
        self.summaries = self.get_summaries()

    def set_additional_options(self):
        pass

    def option_variable(self, default, dtype, name):
        with tf.variable_scope("%s_opt_vars" % self.name):
            var = tf.Variable(default, dtype=dtype, name=name, trainable=False, collections=['NETWORK_OPTIONS', tf.GraphKeys.GLOBAL_VARIABLES])
            var_placeholder = tf.placeholder(dtype, var.shape, name="p_%s"%name)

            self.__opt_assign_ops[var] = tf.assign(var, var_placeholder, name="a_%s"%name)
        return var

    def set_option_variable(self, var, value, session):
        assert isinstance(session, tf.Session)
        tensor = self.__opt_assign_ops[var]
        return session.run(tensor, feed_dict={tensor.op.inputs[1]:value})

    @run_once
    @abc.abstractmethod
    def prepare_data(self, sess):
        pass

    @run_once
    @abc.abstractmethod
    def build(self):
        pass

    @run_once
    @abc.abstractmethod
    def get_train_op(self):
        with tf.variable_scope(self.name):
            increment_step = tf.assign_add(self.global_step, 1)
            with tf.control_dependencies([increment_step]):
                with tf.control_dependencies([self.lr_decay]):
                    train_op = tf.Print(self.global_step, [self.global_step], message="Step: ")
        return train_op

    @run_once
    @abc.abstractmethod
    def get_summaries(self):
        with tf.variable_scope(self.name):
            summaries = tf.summary.scalar("foo", 1)
        return summaries

    @abc.abstractmethod
    def lr_decay(self):
        with tf.variable_scope(self.name):
            learning_rate = tf.assign(self.learning_rate, tf.identity(self.learning_rate))
        return learning_rate

    def get_end_of_training_policy(self):
        self.max_iter = self.option_variable(1000, tf.int32, name='max_iter')
        self.end_of_training_policy = \
            tf.cond(tf.less(self.global_step, self.max_iter), lambda: tf.constant(False), lambda: tf.constant(True))
        return self.end_of_training_policy

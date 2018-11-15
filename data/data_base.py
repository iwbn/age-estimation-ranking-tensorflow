import abc
import tensorflow as tf

class Data:
    __metaclass__ = abc.ABCMeta

    # NOT YET IMPLEMENTED. YOU MIGHT LATER IMPLEMENT THIS IF NEEDED.
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def initialize(self, *args):
        pass

    @abc.abstractmethod
    def get_all_datasets(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def runtime_augmentation(dataset, *args):
        pass

    @staticmethod
    @abc.abstractmethod
    def test_time_augmentation(dataset, *args):
        pass


    # function source from https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    @staticmethod
    def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
        """Distort the color of a Tensor image.
        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.
        Args:
          image: 3-D Tensor containing single image in [0, 1].
          color_ordering: Python int, a type of distortion (valid values: 0-3).
          fast_mode: Avoids slower ops (random_hue and random_contrast)
          scope: Optional scope for name_scope.
        Returns:
          3-D Tensor color-distorted image on range [0, 1]
        Raises:
          ValueError: if color_ordering not in [0, 3]
        """
        with tf.name_scope(scope, 'distort_color', [image]):
            if fast_mode:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                else:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                elif color_ordering == 1:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                elif color_ordering == 2:
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                elif color_ordering == 3:
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                else:
                    raise ValueError('color_ordering must be in [0, 3]')

            # The random_* ops do not necessarily clamp.
            return image
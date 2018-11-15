"""
Author: Woobin Im (iwbn@kaist.ac.kr)
"""
import tensorflow as tf


def sum_squared_error(x, y, name=None, parallel_iterations=100, swap_memory=False):
    with tf.name_scope(name or "sse"):
        x_s = tf.shape(x)
        y_s = tf.shape(y)
        out_size = x_s[0] * y_s[0]

        def elem_wise_op(idx, output_ta):
            i, j = tf.div(idx, y_s[0]), tf.mod(idx, y_s[0])

            # if this block is executed on GPU, it occurs error; maybe fixed in later versions > 1.12.0 of Tensorflow
            with tf.device("/CPU:0"):
                x_, y_ = x[i], y[j]
            diff = tf.subtract(x_, y_)
            output = tf.reduce_sum(tf.square(diff))
            output_ta = output_ta.write(idx, output)
            return idx + 1, output_ta

        idx = tf.constant(0, dtype=tf.int32)
        output_ta = tf.TensorArray(dtype=x.dtype, size=out_size)

        _, outputs_ta = tf.while_loop(
            cond=lambda idx, *_: tf.less(idx, out_size),
            body=elem_wise_op,
            loop_vars=(idx, output_ta),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        gathered = outputs_ta.gather(tf.range(0, out_size))
        reshaped = tf.reshape(gathered, [x_s[0], y_s[0]])
        return reshaped


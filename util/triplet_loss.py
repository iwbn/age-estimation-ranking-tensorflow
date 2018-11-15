import tensorflow as tf
import util.Distance

def _cart_by_mat(p, n):
    p_shape = tf.shape(p)
    n_shape = tf.shape(n)

    p_ = tf.expand_dims(p, axis=1)
    n_ = tf.expand_dims(n, axis=2)

    p_ = tf.tile(p_, multiples=[1, n_shape[1], 1])
    n_ = tf.tile(n_, multiples=[1, 1, p_shape[1]])

    p_ = tf.transpose(p_, [0, 2, 1])
    n_ = tf.transpose(n_, [0, 2, 1])

    return p_, n_


def calculate(dist_mat, aff_mat, coef_func=tf.identity, is_coef=tf.constant(True), epsilon=0.1):
    p_, n_ = _cart_by_mat(aff_mat, aff_mat)
    w = tf.where(tf.less(p_ - n_, 0))
    wt = tf.transpose(w)
    w1 = tf.logical_and(tf.not_equal(wt[0], wt[1]), tf.not_equal(wt[0], wt[2]))

    w = tf.boolean_mask(w, w1)
    num_triplets = tf.shape(w)[0]
    ap_cord = w[:,0:2]
    an_cord = tf.gather(w, [0,2], axis=1)

    p = tf.gather_nd(dist_mat, ap_cord)
    n = tf.gather_nd(dist_mat, an_cord)
    p_aff = tf.gather_nd(aff_mat, ap_cord)

    combined = tf.transpose(tf.stack([p,n]))
    labels = tf.tile(tf.constant([[0,1]], dtype=tf.float32), [num_triplets, 1])
    triplet_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=combined, labels=labels)

    triplet_loss = triplet_loss * tf.cond(is_coef, lambda: coef_func((1.0+epsilon)/(p_aff+epsilon)), lambda: 1.)

    return tf.reduce_mean(triplet_loss)


def make_gt_dist_mat(ground):
    with tf.device("/CPU:0"), tf.name_scope("make_gt_dist_mat"):
        gt_idx = ground
        gt_1 = tf.tile(gt_idx, [1, tf.shape(gt_idx)[0]])
        gt_2 = tf.transpose(gt_1, [1, 0])

        dist_gt = tf.abs(gt_1 - gt_2)
    return dist_gt

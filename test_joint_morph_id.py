from model.joint import AgeModelMorph
import tensorflow as tf
import sys, os
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir', "path-to-log", 'Log is saved to this directory.')
flags.DEFINE_integer('gpu', 4, 'GPU to use.')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % FLAGS.gpu


def test_():
    model = AgeModelMorph()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print ("Cannot find any checkpoints in log directory.")
        return
    acc, o_acc, res, dex_res = model.get_test_accuracy()

    model.prepare_data(sess)
    test_iter = model.make_test_iterators()
    sess.run([test_iter.initializer])
    test_next = test_iter.get_next()

    num_correct = 0
    num_one_off = 0
    sae = 0.
    sae_dex = 0.
    num_test = 0
    while True:
        try:
            x, y = sess.run(test_next)
        except tf.errors.OutOfRangeError:
            break
        [accuracy, one_off_accuracy, c, d] = sess.run([acc, o_acc, res, dex_res],
                                          feed_dict={model.imgs: x, model.ages: y, model.phase: "test", model.is_training: False})
        c += 16
        num_correct += accuracy
        sae += float(abs(c-y[0]))
        sae_dex += abs(d-float(y[0]))
        num_one_off += one_off_accuracy
        num_test += 1
        print ("mae: %.4f, mae(dex): %.4f" % (sae/num_test, sae_dex/num_test))
    with open(os.path.join(FLAGS.log_dir,'test-%d.txt'%sess.run(model.global_step)), 'w') as f:
        f.write("mae: %.4f mae(dex): %.4f" % (sae / num_test, sae_dex / num_test))
        print (float(num_correct)/num_test, float(num_one_off)/num_test)
    sess.close()

test_()

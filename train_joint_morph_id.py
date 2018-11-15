from model.joint import AgeModelMorph
import tensorflow as tf
import sys, os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('fold_number', 0, 'Morph fold number.')
flags.DEFINE_string('log_dir', "path-to-log", 'Log is saved to this directory.')
flags.DEFINE_string('loss_weights', "1.0,0.01", 'triplet,xentropy')
flags.DEFINE_float('max_iter', 30000, 'max step')
flags.DEFINE_float('seed', 0, 'random seed.')
flags.DEFINE_float('learning_rate', 5e-4, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 0., 'Lambda value for l2 decay.')
flags.DEFINE_integer('batch_size', 64, 'batch_size.')
flags.DEFINE_bool('use_pretrain', True, 'use pretrain.')
flags.DEFINE_integer('gpu', 2, 'GPU to use.')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % FLAGS.gpu

print (os.environ["CUDA_VISIBLE_DEVICES"])


def train():
    model = AgeModelMorph()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    elif FLAGS.use_pretrain:
        model.load_pretrained(sess)

    model.set_option_variable(model.batch_size, FLAGS.batch_size, sess)
    model.set_option_variable(model.max_iter, FLAGS.max_iter, sess)
    model.set_option_variable(model.fold_num, FLAGS.fold_number, sess)
    model.set_option_variable(model.l2_decay, FLAGS.weight_decay, sess)
    model.set_option_variable(model.exponential_decay, 0.99, sess)
    model.set_option_variable(model.learning_rate, FLAGS.learning_rate, sess)
    model.set_option_variable(model.triplet_weight, float(FLAGS.loss_weights.split(',')[0]), sess)
    model.set_option_variable(model.xentropy_weight, float(FLAGS.loss_weights.split(',')[1]), sess)

    model.prepare_data(sess)
    train_iter, val_iter = model.make_train_iterators()
    sess.run([train_iter.initializer, val_iter.initializer])
    train_next, val_next = train_iter.get_next(), val_iter.get_next()


    while not sess.run(model.end_of_training_policy):
        x, y = sess.run(train_next)
        [_, summaries, step] = sess.run([model.train_op,
                                         model.summaries, model.global_step],
                                         feed_dict={model.imgs: x, model.ages: y})
        if step % 5 == 0:
            train_writer.add_summary(summaries, step)
        if step % 2 == 0:
            x, y = sess.run(val_next)

            [summaries, step] = sess.run([model.summaries, model.global_step],
                                              feed_dict={model.imgs: x, model.ages: y,
                                                         model.phase: "val", model.is_training: False})
            val_writer.add_summary(summaries, step)
            print ("step: %d"%step)
        if step % 500 == 0:
            saver.save(sess, FLAGS.log_dir + '/ckpt' , step)
    saver.save(sess, FLAGS.log_dir + '/ckpt', sess.run(model.global_step))

    sess.close()

train()
import tensorflow as tf
from .model import Model
from data.morph import MorphByIdentity
from pretrained.facenet import FaceNet
import util.Distance
import util.triplet_loss


class AgeModelMorph(Model):
    def __init__(self):
        Model.__init__(self, 'joint_model')
        self.end_of_training_policy = self.get_end_of_training_policy()

    def set_additional_options(self):
        self.end_of_training_policy = self.get_end_of_training_policy()
        self.l2_decay = self.option_variable(1e-4, tf.float32, name='l2_decay')
        self.keep_prob = self.option_variable(0.5, tf.float32, name='keep_prob')
        self.batch_size = self.option_variable(36, tf.int64, name='batch_size')
        self.phase = self.option_variable("train", tf.string, name='model_phase')
        self.fold_num = self.option_variable(-1, tf.int32, name='fold_num')
        self.exponential_decay = self.option_variable(0.999, tf.float32, name='exponential_decay')
        self.triplet_weight = self.option_variable(0.9, tf.float32, "triplet_weight")
        self.xentropy_weight = self.option_variable(0.1, tf.float32, "xentropy_weight")
        self.min_age = tf.constant(16, name='min_age')
        self.max_age = tf.constant(77, name='max_age')
        self.data_seed = self.option_variable(0, tf.int32, name='data_seed')

    def prepare_data(self, sess):
        self.morph = MorphByIdentity('Morph')
        self.morph.initialize(sess, self.data_seed, self.fold_num)
        self.train_set, self.val_set, self.test_set = self.morph.get_all_datasets()

    def calc_triplet_loss(self, age, age_max):
        age = tf.cast(tf.expand_dims(age, -1), tf.float32)
        age_max = tf.cast(age_max, tf.float32)

        dist = util.Distance.sum_squared_error(self.feat_norm, self.feat_norm,
                                               parallel_iterations=100, swap_memory=False)
        aff = util.triplet_loss.make_gt_dist_mat(age / age_max)
        aff_ = tf.cast(aff, tf.float32)

        self.tri_loss = util.triplet_loss.calculate(dist, aff_,
                                                    lambda x: tf.pow(x, 1.),
                                                    is_coef=tf.constant(True),
                                                    epsilon=0.1)

    def build(self):
        self.pretrained = FaceNet()
        self.pretrained.opts.weight_path = \
            'pretrained/FaceNet/20170216-091149/model-20170216-091149.ckpt-250000.npy'
        self.imgs = tf.placeholder(dtype=tf.float32, shape=self.pretrained.input_shape, name='input_images')
        self.ages = tf.placeholder(dtype=tf.int32, shape=[None], name='age_labels_in_num')
        self.age_one_hot = tf.one_hot(self.ages-self.min_age, self.max_age-self.min_age+1, axis=1, name='age_labels_in_one_hot')
        self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')

        self.bottleneck = self.pretrained.construct(self.imgs, self.is_training, self.keep_prob)
        self.feat_norm = tf.nn.l2_normalize(self.bottleneck, axis=1)

        self.calc_triplet_loss(self.ages-self.min_age, self.max_age-self.min_age)

        self.fc1 = self.fc_bn(self.feat_norm, 128, activation=tf.nn.relu, name='fc1',
                                   kernel_regularizer=lambda w: tf.nn.l2_loss(w))
        self.fc2 = tf.layers.dense(self.fc1, 62, activation=None, name='fc2',
                                   kernel_regularizer=lambda w: tf.nn.l2_loss(w))
        self.softmax = tf.nn.softmax(self.fc2)
        self.xentropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.age_one_hot, logits=self.fc2))
        self.l2_reg_loss = tf.multiply(self.l2_decay/2., tf.losses.get_regularization_loss())

        self.predicted_class = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)
        self.target_class = tf.cast(self.ages-self.min_age, tf.int32)
        self.mae = tf.reduce_mean(tf.cast(tf.abs(self.predicted_class-self.target_class), tf.float32))
        self.num_correct = tf.reduce_sum(tf.cast(tf.equal(self.predicted_class, self.target_class), tf.float32))
        one_off_correct = tf.cast(tf.less_equal(tf.abs(self.predicted_class - self.target_class), 1), tf.float32)
        self.num_one_off_correct = tf.reduce_sum(one_off_correct)

        self.accuracy = tf.divide(self.num_correct,
                                  tf.cast(tf.shape(self.predicted_class)[0], tf.float32))

        self.one_off_accuracy = tf.divide(self.num_one_off_correct,
                                          tf.cast(tf.shape(self.predicted_class)[0], tf.float32))
        self.train_op = self.get_train_op()

    def get_test_accuracy(self):
        # input must include 1 instance augmented with crops
        out_response = tf.reduce_mean(self.fc2, axis=0, keepdims=True)
        out_response = tf.nn.softmax(out_response)
        gt = tf.reduce_mean(self.ages-self.min_age, axis=0, keepdims=True)
        predicted_class_dex = tf.reduce_sum(tf.cast(tf.range(self.min_age-1, self.max_age), tf.float32) * tf.squeeze(out_response))
        predicted_class = tf.cast(tf.argmax(out_response, axis=1), tf.int32)
        target_class = tf.cast(gt, tf.int32)
        num_correct = tf.reduce_sum(tf.cast(tf.equal(predicted_class, target_class), tf.float32))
        one_off_correct = tf.cast(tf.less_equal(tf.abs(predicted_class - target_class), 1), tf.float32)
        num_one_off_correct = tf.reduce_sum(one_off_correct)
        
        accuracy = tf.divide(num_correct,
                                  tf.cast(tf.shape(predicted_class)[0], tf.float32))

        one_off_accuracy = tf.divide(num_one_off_correct,
                                          tf.cast(tf.shape(predicted_class)[0], tf.float32))
        mae = tf.reduce_mean(tf.cast(tf.abs(self.predicted_class - self.target_class), tf.float32))
        
        return accuracy, one_off_accuracy, predicted_class, predicted_class_dex

    def make_train_iterators(self):
        train_iter = self.morph.runtime_augmentation(self.train_set,
                                                 self.pretrained.input_shape[1:3],
                                                 rand_crop=True,
                                                 rand_flip=True,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)
        val_iter = self.morph.runtime_augmentation(self.val_set,
                                              self.pretrained.input_shape[1:3],
                                              rand_crop=False,
                                              rand_flip=False,
                                              batch_size=self.batch_size,
                                              shuffle=True)
        return train_iter, val_iter

    def make_test_iterators(self):
        test_iter = self.morph.test_time_augmentation(self.test_set, self.pretrained.input_shape[1:3])
        return test_iter

    def get_summaries(self):
        tr = []
        with tf.variable_scope(self.name):
            tr.append(tf.summary.scalar("learning_rate", self.ongoing_learning_rate))
            tr.append(tf.summary.scalar("accuracy", self.accuracy))
            tr.append(tf.summary.scalar("mae", self.mae))
            tr.append(tf.summary.scalar("one_off_accuracy", self.one_off_accuracy))
            tr.append(tf.summary.scalar("loss", self.loss))
            tr.append(tf.summary.scalar("l2_regularization_loss", self.l2_reg_loss))
            tr.append(tf.summary.scalar("xentropy_loss", self.xentropy_loss))
            tr.append(tf.summary.scalar("triplet_loss", self.tri_loss))

            def assign(acc_var, d_acc_var, mae_var, d_mae_var):
                with tf.control_dependencies([self.ema.apply([acc_var, d_acc_var, mae_var, d_mae_var])]):
                    with tf.control_dependencies([tf.assign(acc_var, self.accuracy),
                                                  tf.assign(d_acc_var,
                                                            self.accuracy - self.ema.average(acc_var)),
                                                  tf.assign(mae_var, self.mae),
                                                  tf.assign(d_mae_var,
                                                            self.mae - self.ema.average(mae_var))
                                                  ]):
                        ret = (tf.identity(self.ema.average(acc_var)), tf.identity(self.ema.average(d_acc_var)),
                                tf.identity(self.ema.average(mae_var)), tf.identity(self.ema.average(d_mae_var)))
                        return ret

            cond_val_test = lambda: tf.cond(tf.equal(self.phase, "val"),
                                            lambda: assign(self.tv_acc, self.d_tv_acc, self.tv_mae, self.d_tv_mae),
                                            lambda: assign(self.ts_acc, self.d_ts_acc, self.ts_mae, self.d_ts_mae))
            moving_accuracy = tf.cond(tf.equal(self.phase, "train"),
                                      lambda: (self.ema.average(self.tr_acc), self.ema.average(self.d_tr_acc),
                                               self.ema.average(self.tr_mae), self.ema.average(self.d_tr_mae)),
                                      lambda: cond_val_test())
            tr.append(tf.summary.scalar("moving_accuracy", moving_accuracy[0]))
            tr.append(tf.summary.scalar("d_moving_accuracy", moving_accuracy[1]))
            tr.append(tf.summary.scalar("moving_mae", moving_accuracy[2]))
            tr.append(tf.summary.scalar("d_moving_mae", moving_accuracy[3]))
            tr.append(tf.summary.scalar("loss_moving_var", self.ema.average(self.moving_var)))
        return tf.summary.merge(tr)

    def get_train_op(self):
        with tf.variable_scope(self.name):
            self.tr_acc = tf.Variable(0, dtype=tf.float32, name='train_accuracy')
            self.tv_acc = tf.Variable(0, dtype=tf.float32, name='val_accuracy')
            self.ts_acc = tf.Variable(0, dtype=tf.float32, name='test_accuracy')
            self.d_tr_acc = tf.Variable(0, dtype=tf.float32, name='d_train_accuracy')
            self.d_tv_acc = tf.Variable(0, dtype=tf.float32, name='d_val_accuracy')
            self.d_ts_acc = tf.Variable(0, dtype=tf.float32, name='d_test_accuracy')
            self.tr_mae = tf.Variable(0, dtype=tf.float32, name='train_mae')
            self.tv_mae = tf.Variable(0, dtype=tf.float32, name='val_mae')
            self.ts_mae = tf.Variable(0, dtype=tf.float32, name='test_mae')
            self.d_tr_mae = tf.Variable(0, dtype=tf.float32, name='d_train_mae')
            self.d_tv_mae = tf.Variable(0, dtype=tf.float32, name='d_val_mae')
            self.d_ts_mae = tf.Variable(0, dtype=tf.float32, name='d_test_mae')

            self.ema = tf.train.ExponentialMovingAverage(self.exponential_decay)
            self.loss = self.l2_reg_loss + self.tri_loss * self.triplet_weight + \
                        self.xentropy_loss * self.xentropy_weight

            all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            fc_vars = [var for var in all_vars if var.name.startswith('fc')]
            other_vars = list(set(all_vars) - set(fc_vars))

            increment_step = tf.assign(self.global_step, self.global_step+1)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.lr_decay()
            with tf.control_dependencies([increment_step]+extra_update_ops):
                optimize = tf.train.AdamOptimizer(
                    learning_rate=self.ongoing_learning_rate
                ).minimize(self.loss, var_list=other_vars)
                fc_optimize = tf.train.AdamOptimizer(
                    learning_rate=self.ongoing_learning_rate
                ).minimize(self.xentropy_loss, var_list=fc_vars)

            with tf.control_dependencies([self.ema.apply([self.loss])]):
                self.moving_var = tf.square(self.loss - self.ema.average(self.loss))
            with tf.control_dependencies([optimize, fc_optimize,
                                          self.ema.apply([self.tr_acc, self.d_tr_acc,
                                                          self.tr_mae, self.d_tr_mae,
                                                          self.moving_var])]):
                with tf.control_dependencies([tf.assign(self.tr_acc, self.accuracy),
                                              tf.assign(self.d_tr_acc, self.accuracy - self.ema.average(self.tr_acc)),
                                              tf.assign(self.tr_mae, self.mae),
                                              tf.assign(self.d_tr_mae, self.mae - self.ema.average(self.tr_mae))
                                              ]):
                    train_op = tf.constant(True)
            return train_op

    def lr_decay(self):
        with tf.variable_scope(self.name):
            self.ongoing_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                       2000, 0.96, staircase=True)
        return self.ongoing_learning_rate

    def get_end_of_training_policy(self):
        self.max_iter = self.option_variable(1000, tf.int32, name='max_iter')
        self.min_iter = self.option_variable(30, tf.int32, name='min_iter')

        self.end_of_training_policy = tf.greater(self.global_step, self.max_iter)
        return self.end_of_training_policy

    def fc_bn(self, input, dim, activation, name, kernel_regularizer=None):
        fc1 = tf.layers.dense(input, dim, name=name, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=kernel_regularizer)
        fc1_bn = tf.layers.batch_normalization(
                    inputs=fc1,
                    axis=-1,
                    momentum=0.9,
                    epsilon=0.001,
                    center=True,
                    scale=True,
                    training = self.is_training,
                    name=name+'_bn'
                )
        return activation(fc1_bn, name=name+'_a')

    def load_pretrained(self, sess):
        self.pretrained.load_pretrained(sess, [''], self.pretrained.opts.weight_path)

import tensorflow as tf
import sys
sys.path.append("../")
import os
import numpy as np
from data.data_base import Data
import pathconf

BASE_PATH = pathconf.get_path('MorphBasePath')
CSV_PATH = 'morph_2008_nonCommercial.csv'

_p = lambda p : os.path.join(BASE_PATH, p)
_tp = lambda p : tf.reduce_join([BASE_PATH, p])


class Morph(Data):
    def initialize(self, sess, seed, fold):
        seed = int(sess.run(seed))

        # Loads the csv
        with open(_p(CSV_PATH), 'r') as f:
            header_row = f.readline()
            headers = header_row.strip().split(',')
            meta = {i: [] for i in headers}
            for l in f:
                l = l.strip()
                data = l.split(',')
                if len(data) != len(headers):
                    print("inconsistent row: %s" % l)
                    continue
                for i in range(len(headers)):
                    meta[headers[i]].append(data[i])
        self.num_data = len(meta[headers[0]])
        self.headers = headers
        self.meta = meta

        all_idx = list(range(self.num_data))
        np.random.seed(seed)
        np.random.shuffle(all_idx)
        num_in_one_fold = int(self.num_data*0.2)
        fold_indices = []
        for i in range(5):
            fold_indices.append(all_idx[i*num_in_one_fold:(i+1)*num_in_one_fold])
        else:
            fold_indices[-1].extend(all_idx[(i+1)*num_in_one_fold: self.num_data])
        print (num_in_one_fold)

        tr_idx = []
        ts_idx = []
        for i, fold_idx in enumerate(fold_indices):
            if i == sess.run(fold):
                ts_idx.extend(fold_idx)
            else:
                tr_idx.extend(fold_idx)
        np.random.seed(seed+100)
        np.random.shuffle(tr_idx)
        num_val_set = int(len(tr_idx)*0.1)
        val_idx = tr_idx[0:num_val_set]
        tr_idx = tr_idx[num_val_set:]

        def make_dataset(indices):
            get_from_py = lambda idx: (_p(meta['photo'][idx].replace('Album2', 'Album2_aligned')), np.int64(meta['age'][idx]))

            def _parse_function(filename, label):
                image_string = tf.read_file(filename)
                image_decoded = tf.image.decode_image(image_string)
                rgb = tf.cond(tf.equal(tf.shape(image_decoded)[2], 1),
                              lambda: tf.tile(image_decoded, [1, 1, 3]),
                              lambda: tf.identity(image_decoded))
                return rgb, label

            dataset = tf.data.Dataset.from_tensor_slices(indices)
            dataset = dataset.shuffle(6000)
            dataset = dataset.map(lambda idx: tf.py_func(get_from_py, [idx], [tf.string, tf.int64]))
            dataset = dataset.map(_parse_function, num_parallel_calls=1)
            return dataset
        self.train_set = make_dataset(tr_idx)
        self.val_set = make_dataset(val_idx)
        self.test_set = make_dataset(ts_idx)

    def get_all_datasets(self):
        return self.train_set, self.val_set, self.test_set

    @staticmethod
    def runtime_augmentation(dataset, output_size, rand_crop, rand_flip, batch_size, shuffle, distort_color=True,
                             num_epoch=None):
        if rand_crop:
            scale = 1.05
            crop_resize = (int(output_size[1] * scale), int(output_size[0] * scale))
            dataset = dataset.map(lambda x, y: (tf.image.resize_bicubic(tf.expand_dims(x, 0), crop_resize)[0], y))
            dataset = dataset.map(lambda x, y: (tf.random_crop(x, (output_size[1], output_size[0], 3)), y))
        else:
            dataset = dataset.map(lambda x, y: (tf.image.resize_bicubic(tf.expand_dims(x, 0), output_size)[0], y))
        if rand_flip:
            dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
        if distort_color:
            dataset = dataset.map(lambda x, y: (Data.distort_color(x), y))
        dataset = dataset.repeat(num_epoch)
        # if shuffle:
        #    dataset = dataset.shuffle(6000)
        dataset = dataset.prefetch(256)
        dataset = dataset.batch(batch_size)
        return dataset.make_initializable_iterator()

    @staticmethod
    def test_time_augmentation(dataset, output_size):
        boxes = tf.constant([[0, 0, 0.9, 0.9],
                             [0, 0.1, 0.9, 1],
                             [0.1, 0, 1.0, 0.9],
                             [0.1, 0.1, 1.0, 1.0],
                             [0.05, 0.05, 0.95, 0.95]])
        indices = tf.zeros((5), dtype=tf.int32)
        testset = dataset.map(lambda x, y:
                              (tf.image.crop_and_resize(tf.expand_dims(x, 0),
                                                        boxes, indices,
                                                        [output_size[1], output_size[0]])
                               , y))
        testset = testset.map(lambda x, y:
                              (tf.concat([x,
                                          tf.gather(x, tf.range(output_size[1] - 1, -1, -1), axis=2)
                                          # filpping
                                          ], axis=0), tf.ones(10, dtype=tf.int64) * y))
        testset = testset.prefetch(256)
        return testset.make_initializable_iterator()


class MorphByIdentity(Morph):
    def initialize(self, sess, seed, fold):
        seed = int(sess.run(seed))
        indices_id = {}

        # Loads the csv
        with open(_p(CSV_PATH), 'r') as f:
            header_row = f.readline()
            headers = header_row.strip().split(',')
            meta = {i: [] for i in headers}
            for idx, l in enumerate(f):
                l = l.strip()
                data = l.split(',')
                if len(data) != len(headers):
                    print("inconsistent row: %s" % l)
                    continue
                for i in range(len(headers)):
                    meta[headers[i]].append(data[i])
                try:
                    indices_id[int(meta['id_num'][idx])].add(idx)
                except KeyError:
                    indices_id[int(meta['id_num'][idx])] = {idx}
        self.num_ids = len(indices_id)
        self.num_data = len(meta[headers[0]])
        self.headers = headers
        self.meta = meta

        all_idx = list(indices_id.keys())
        np.random.seed(seed)
        np.random.shuffle(all_idx)
        num_in_one_fold = int(self.num_ids*0.2)
        fold_indices = []
        for i in range(5):
            fold_indices.append(all_idx[i*num_in_one_fold:(i+1)*num_in_one_fold])
        else:
            fold_indices[-1].extend(all_idx[(i+1)*num_in_one_fold: self.num_data])
        print (num_in_one_fold)

        tr_idx = []
        ts_idx = []
        for i, fold_idx in enumerate(fold_indices):
            if i == sess.run(fold):
                for fo in fold_idx:
                    ts_idx.extend(indices_id[fo])
            else:
                tr_idx.extend(fold_idx)
        np.random.seed(seed+100)
        np.random.shuffle(tr_idx)
        num_val_set = int(len(tr_idx)*0.1)
        val_idx_ = tr_idx[0:num_val_set]
        tr_idx_ = tr_idx[num_val_set:]
        tr_idx, val_idx = [], []
        for fo in val_idx_:
            val_idx.extend(indices_id[fo])
        for fo in tr_idx_:
            tr_idx.extend(indices_id[fo])

        def make_dataset(indices, shuffle=True):
            get_from_py = lambda idx: (_p(meta['photo'][idx].replace('Album2', 'Album2_aligned')), np.int64(meta['age'][idx]))

            def _parse_function(filename, label):
                image_string = tf.read_file(filename)
                image_decoded = tf.image.decode_image(image_string)
                rgb = tf.cond(tf.equal(tf.shape(image_decoded)[2], 1),
                              lambda: tf.tile(image_decoded, [1, 1, 3]),
                              lambda: tf.identity(image_decoded))
                return rgb, label

            dataset = tf.data.Dataset.from_tensor_slices(indices)
            if shuffle:
                dataset = dataset.shuffle(6000)
            dataset = dataset.map(lambda idx: tf.py_func(get_from_py, [idx], [tf.string, tf.int64]))
            dataset = dataset.map(_parse_function, num_parallel_calls=1)
            return dataset
        self.train_set = make_dataset(tr_idx)
        self.val_set = make_dataset(val_idx)
        self.test_set = make_dataset(ts_idx, shuffle=False)

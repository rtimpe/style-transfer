import os

import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from imageio import imread
import skimage.transform

def make_decoder(inputs, layers, sess):
    all_params = []
    all_regularized = []
    prev_layer = inputs
    with tf.variable_scope('decoder'):
        for (i, layer) in enumerate(layers):
            if layer == 'upsample':
                (height, width) = prev_layer.get_shape()[1:3]
                out = tf.image.resize_images(prev_layer, (int(height) * 2, int(width) * 2),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                prev_layer = out
            else:
                (filter_size, num_filters) = layer
                (params, conv_out, regularized) = conv_layer(i, prev_layer.shape[3],
                                                             num_filters, filter_size, prev_layer)
                prev_layer = conv_out
                all_params += params
                all_regularized += regularized

        init = tf.variables_initializer(all_params)
        sess.run(init)

    return prev_layer, all_regularized

# https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca#11336203
def whiten(X, fudge=1E-18):

    # get the covariance matrix
    Xcov = np.dot(X, X.T) / float(X.shape[1] - 1)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # Xcov should be psd, but numerical issues can lead to negative eigenvalues
    d[d < 0] = 0

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d+fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(W, X)

    return X_white, W

def color(C, S):
    Scov = np.dot(S, S.T) / float(S.shape[1] - 1)

    d, V = np.linalg.eigh(Scov)

    # Scov should be psd, but numerical issues can lead to negative eigenvalues
    d[d < 0] = 0
    D = np.diag(np.sqrt(d))

    cm = np.dot(np.dot(V, D), V.T)

    CS = np.dot(cm, C)

    return CS, cm

def encode_file(fname, layer):
    img = imread(fname)
    img = skimage.transform.resize(img, (224, 224, 3))
    img = img[np.newaxis, :, :, :]
    img = img * 255.0
    encoded_img = encode_image(img, 'vgg_19/' + layer + '/' + layer + '_1')
    return encoded_img

def encode_image(image, layer_name):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            images_ph = tf.placeholder('float', (None, 224, 224, 3))
            _, d = make_encoder(images_ph, sess)
            encoded = d[layer_name]
            encoded_img = sess.run(encoded, feed_dict={images_ph: image})
            return encoded_img

def make_encoder(input_ph, sess, mean_img=[123.68, 116.779, 103.939]):
    mean_img = np.array(mean_img, dtype=np.float32)[np.newaxis, np.newaxis, :] # for some reason tensorflow can't do this automatically
    mean_img = tf.tile(mean_img, (224, 224, 1))
    preprocessed = input_ph - mean_img
    # preprocessed = tf.Print(preprocessed, [preprocessed], summarize=10)
    vgg = nets.vgg.vgg_19(preprocessed)
    restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_19')
    saver = tf.train.Saver(var_list=restore_vars)
    saver.restore(sess, 'vgg_19.ckpt')
    return vgg

def make_dataset(data_dir, batch_size=64, include_filenames=False):
    filenames = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]
    filenames = tf.constant(filenames)
    def _parse_function(filename):
        image = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [224, 224],
                                               method=tf.image.ResizeMethod.BILINEAR)
        if include_filenames:
            return image_resized, filename
        return image_resized

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function)

    return dataset

def load_wrapper(filename):
    return np.load(filename.decode('utf-8'))

def make_precomputed_dataset(data_dir, features_name, batch_size=64, num_images=1000):
    image_filenames = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]
    image_filenames = tf.constant(image_filenames)
    features_dir = os.path.join(data_dir, features_name)
    features_filenames = [os.path.join(features_dir, os.path.splitext(img_name)[0]) + '.npy'
                          for img_name in os.listdir(data_dir)]
    features_filenames = tf.constant(features_filenames)
    def _parse_function(image_filename, features_filename):
        features = tf.py_func(load_wrapper, [features_filename], tf.float32)

        image = tf.read_file(image_filename)
        image_decoded = tf.image.decode_jpeg(image, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [224, 224],
                                               method=tf.image.ResizeMethod.BILINEAR)
        return (image_resized, features)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, features_filenames))
    if num_images is not None:
        dataset = dataset.take(num_images)
    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(50)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 2)

    return dataset

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def compute_features(data_dir, batch_size=32, out_layer_name='vgg_19/conv5/conv5_1', num_images=1000):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_ph = tf.placeholder('float', (None, 224, 224, 3))
            _, layers = make_encoder(input_ph, sess)
            out_layer = layers[out_layer_name]
            dataset = make_dataset(data_dir, include_filenames=True)
            if num_images is not None:
                dataset = dataset.take(num_images)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()

            path = os.path.join(data_dir, out_layer_name.split('/')[1])
            os.makedirs(path, exist_ok=True)

            while True:
                try:
                    batch, filenames = sess.run(next_batch)
                    features_batch = sess.run(out_layer, feed_dict={input_ph: batch})

                    for (features, filename) in zip(features_batch, filenames):
                        filename = os.path.basename(str(filename))
                        filename = os.path.splitext(filename)[0]
                        np.save(os.path.join(path, filename), features)
                except tf.errors.OutOfRangeError:
                    break

def conv_layer(i, in_size, num_filters, filter_size, input_layer):
    with tf.variable_scope('conv_layer_' + str(i)):
        initializer = tf.contrib.layers.xavier_initializer()
        filters = tf.Variable(initializer([filter_size, filter_size, int(in_size), num_filters]),
                              name='filters', dtype=tf.float32)
        tf.summary.histogram('filters', filters)
        b = tf.Variable(tf.zeros([num_filters]), name='biases')
        tf.summary.histogram('biases', b)

        pre_activation = tf.nn.conv2d(input_layer, filters, [1, 1, 1, 1], "SAME") + b
        conv_out = tf.nn.relu(pre_activation)

        tf.summary.histogram('pre_activations', pre_activation)
        tf.summary.histogram('activations', conv_out)

    return [filters, b], conv_out, [filters]

def create_loss(images_ph, features_ph, reconstructed_image, layer_name, regularized, sess, lambda_rl=1, lambda_reg=1e-4):
    img_diff = images_ph - reconstructed_image
    img_loss = tf.reduce_mean(tf.square(tf.reshape(img_diff, [tf.shape(img_diff)[0], -1])), axis=1)
    _, d = make_encoder(reconstructed_image, sess)
    encoded = d[layer_name]

    features_diff = encoded - features_ph
    reconstruction_loss = tf.reduce_mean(tf.square(tf.reshape(features_diff, [tf.shape(features_diff)[0], -1])), axis=1)

    per_img_loss = img_loss + lambda_rl * reconstruction_loss

    regularizer = tf.add_n([tf.nn.l2_loss(r) for r in regularized])
    total_loss = tf.reduce_sum(per_img_loss) + lambda_reg * regularizer

    tf.summary.scalar('loss', total_loss)
    return (per_img_loss, total_loss)

# https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables#35618160
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def setup_training(loss_fn, dataset, sess, lr=1e-4):
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decoder')
    print(trainable_vars)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_fn, var_list=trainable_vars)
    # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss_fn, var_list=trainable_vars)

    merged = tf.summary.merge_all()

    # initialize remaining variables (should just be variables from the optimizer)
    # print(sess.run(tf.report_uninitialized_variables()))
    initialize_uninitialized(sess)

    return (train_step, merged)

def evaluate_dataset(dataset, loss_fn, features_ph, images_ph, sess):
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    total_loss = 0.0
    total_images = 0
    while True:
        try:
            images, features = sess.run(next_batch)
            num_images = images.shape[0]
            loss = sess.run(loss_fn, feed_dict={features_ph: features, images_ph: images})
            total_loss += loss.sum()
            total_images += num_images

        except tf.errors.OutOfRangeError:
            break
    loss = total_loss / total_images
    return loss

def train(loss_fn, train_step, merged, train_dataset, val_dataset, images_ph, features_ph, sess, num_epochs=5, summary_freq=5):
    writer = tf.summary.FileWriter('summaries', sess.graph)

    # iterator = train_dataset.make_one_shot_iterator()
    # next_batch = iterator.get_next()
    # images, features = sess.run(next_batch)
    # (summary, loss) = sess.run([merged, loss_fn], {features_ph: features, images_ph: images})
    # print('initial loss:', loss)
    iters = 0
    loss = evaluate_dataset(val_dataset, loss_fn, features_ph, images_ph, sess)
    print('initial validation loss', loss)
    for epoch in range(num_epochs):
        iterator = train_dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        while True:
            try:
                images, features = sess.run(next_batch)
                sess.run(train_step, feed_dict={features_ph: features, images_ph: images})
                if iters % summary_freq == 0:
                    (summary, loss) = sess.run([merged, loss_fn], {features_ph: features, images_ph: images})
                    writer.add_summary(summary, iters)
                    loss = loss.mean()
                    print('training loss', loss)

                iters += 1

            except tf.errors.OutOfRangeError:
                break
        loss = evaluate_dataset(val_dataset, loss_fn, features_ph, images_ph, sess)
        print('validation loss after epoch', epoch, 'is', loss)

# unused functions that might come in handy later
def compute_features_tfrecord(data_dir, out_file_name, batch_size=32,
                              out_layer_name='vgg_19/conv5/conv5_1', num_images=1000):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_ph = tf.placeholder('float', (None, 224, 224, 3))
            _, layers = make_encoder(input_ph, sess)
            out_layer = layers[out_layer_name]
            dataset = make_dataset(data_dir)
            if num_images is not None:
                dataset = dataset.take(num_images)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()
            with tf.python_io.TFRecordWriter(out_file_name) as writer:
                while True:
                    try:
                        batch = sess.run(next_batch)
                        features_batch = sess.run(out_layer, feed_dict={input_ph: batch})

                        for features in features_batch:
                            example = tf.train.Example(
                                features=tf.train.Features(
                                    feature={
                                        'encoded': _float_feature(np.reshape(features, [-1])),
                                        'shape': _int64_feature(features.shape)
                                    }))
                            writer.write(example.SerializeToString())

                    except tf.errors.OutOfRangeError:
                        break

def make_dataset_from_tfrecord(filenames, shape):
    dataset = tf.data.TFRecordDataset(filenames)
    def _parse_function(example):
        features = {
            'encoded': tf.FixedLenFeature((np.prod(shape)), tf.float32)
        }
        parsed_features = tf.parse_single_example(example, features)
        encoded = tf.reshape(parsed_features['encoded'], shape)
        return encoded

    dataset = dataset.map(_parse_function)
    return dataset


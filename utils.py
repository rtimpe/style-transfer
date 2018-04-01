import tensorflow as tf
import numpy as np

def make_decoder(inputs, layers, sess):
    all_params = []
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
                (params, conv_out) = conv_layer(i, prev_layer.shape[3],
                                                num_filters, filter_size, prev_layer)
                prev_layer = conv_out
                all_params += params

        init = tf.variables_initializer(all_params)
        sess.run(init)

    return prev_layer

def conv_layer(i, in_size, num_filters, filter_size, input_layer):
    with tf.variable_scope('conv_layer_' + str(i)):
        initializer = tf.contrib.layers.xavier_initializer()
        filters = tf.Variable(initializer([filter_size, filter_size, int(in_size), num_filters]),
                              name='filters', dtype=tf.float32)
        tf.summary.histogram('filters', filters)
        b = tf.Variable(tf.zeros([num_filters]), name='biases')
        tf.summary.histogram('biases', b)

        conv_out = tf.nn.relu(tf.nn.conv2d(input_layer, filters, [1, 1, 1, 1], "SAME") + b)

        tf.summary.histogram('activations', conv_out)

    return [filters, b], conv_out

# https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables#35618160
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def train(loss_fn, dataset, input_ph, sess, lr=1e-4, batch_size=16, num_epochs=5):
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decoder')
    print(trainable_vars)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_fn, var_list=trainable_vars)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('summaries', sess.graph)

    # initialize remaining variables (should just be variables from the optimizer)
    print(sess.run(tf.report_uninitialized_variables()))
    initialize_uninitialized(sess)


    dataset = dataset.batch(32)
    dataset = dataset.prefetch(50)

    for i in range(num_epochs):
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        while True:
            try:
                batch = sess.run(next_batch)
                sess.run(train_step, feed_dict={input_ph: batch})

            except tf.errors.OutOfRangeError:
                break
        (summary, loss) = sess.run([merged, loss_fn], feed_dict={input_ph: batch})
        writer.add_summary(summary, i)
        print('loss at epoch', i, ':', loss)


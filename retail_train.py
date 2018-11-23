import os
import math
import utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.models.official.mnist import dataset
# from keras.layers import Conv2D

# tf.app.flags.DEFINE_string(
#   'train_dir', '/tmp/tfmodel/',
#   'Directory where checkpoints and event logs are written to.')
#
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
#
# tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
#                             'Use CPUs to deploy clones.')
file_dir = '/home/fs168/dataSet/crop_person'
batch_size = 100
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
tf.app.flags.DEFINE_integer('display_step', 100, 'display results per d')
tf.app.flags.DEFINE_integer('n_input', 112, 'shape of inputs')
tf.app.flags.DEFINE_integer('n_class', 4, 'Number of classes')
tf.app.flags.DEFINE_integer('training_epochs', 1001, 'Number of epochs')

FLAGS = tf.app.flags.FLAGS

'''define some functions to slim the code'''
def WeightsVariable(shape, name_str=None, stddev=0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str=None, stddev=1e-5):
    initial = tf.random_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

def Conv2d(x, W, b, stride, padding='SAME'):
    y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    y = tf.nn.bias_add(y, b)
    return y

def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID')


def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
        y = activate(y)
    return y

def EvaluateModelOnDataset(sess, images, labels):
    # n_samples = images.shape[0]                        # Num of images
    n_samples = len(images)
    per_batch_size = batch_size
    loss = 0
    acc = 0
    # divide samples to avoid out of Memory
    if n_samples < per_batch_size:
        batch_count = 1
        loss, acc = sess.run([binary_entropy_loss, accuracy], feed_dict={X_origin:images, Y_true:labels,
                                                                        learning_rate:FLAGS.learning_rate})
    else:
        batch_count = int(n_samples / per_batch_size)
        batch_start = 0
        for idx in range(batch_count):
            batch_loss, batch_acc = sess.run([binary_entropy_loss, accuracy], feed_dict={X_origin:images[batch_start:batch_start + per_batch_size],
                                                                                        Y_true:labels[batch_start:batch_start + per_batch_size],
                                                                                         learning_rate:FLAGS.learning_rate})
            batch_start += per_batch_size
            loss += batch_loss
            acc += batch_acc

    return loss / batch_count, acc / batch_count

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda : exp_moving_avg.average(mean), lambda :mean)
    v = tf.cond(is_test, lambda : exp_moving_avg.average(variance), lambda :variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

with tf.Graph().as_default():
    with tf.name_scope('Inputs'):
        X_origin = tf.placeholder(tf.float32, [None, FLAGS.n_input, FLAGS.n_input, 3], name='X_origin')
        Y_true = tf.placeholder(tf.float32, [None, 4], name='Y_true')
        # tst = tf.placeholder(tf.bool)
        # iter_step = tf.placeholder(tf.int32)
        # 784 conver to 28 * 28
        X_image = tf.reshape(X_origin, [-1, 112, 112, 3])
    with tf.name_scope('Inference'):
        # First layer
        with tf.name_scope('Con2d_1'):
            W_conv1_1 = WeightsVariable([3, 3, 3, 64], name_str='weights_conv1_1')
            b_conv1_1 = BiasesVariable(shape=[64], name_str='biases_conv1_1')
            conv1_out = Conv2d(X_image, W_conv1_1, b_conv1_1, stride=1)
            # Y1bn, update_ema1 = batchnorm(conv1_out, tst, iter_step, b_conv1, convolutional=True)
            active1_out = tf.nn.relu(conv1_out)
            W_conv1_2 = WeightsVariable([3, 3, 64, 64], name_str='weights_conv1_2')
            b_conv1_2 = BiasesVariable(shape=[64], name_str='biases_conv1_2')
            conv1_out = Conv2d(active1_out, W_conv1_2, b_conv1_2, stride=1)
            # Y1bn, update_ema1 = batchnorm(conv1_out, tst, iter_step, b_conv1, convolutional=True)
            active1_out = tf.nn.relu(conv1_out)
        with tf.name_scope('Pool2d_1'):
            pool1_out = Pool2d(active1_out)

        # Second layer
        with tf.name_scope('Con2d_2'):
            W_conv2 = WeightsVariable([3, 3, 64, 128], name_str='weights_conv2')
            b_conv2 = BiasesVariable(shape=[128], name_str='biases_conv2')
            conv2_out = Conv2d(pool1_out, W_conv2, b_conv2, stride=1)
            # Y2bn, update_ema2 = batchnorm(conv2_out, tst, iter_step, b_conv2, convolutional=True)
            active2_out = tf.nn.relu(conv2_out)
        with tf.name_scope('Pool2d_2'):
            pool2_out = Pool2d(active2_out)

        with tf.name_scope('Con2d_3'):
            W_conv3 = WeightsVariable([3, 3, 128, 256], name_str='weights_conv3')
            b_conv3 = BiasesVariable(shape=[256], name_str='biases_conv3')
            conv3_out = Conv2d(pool2_out, W_conv3, b_conv3, stride=1)
            # Y2bn, update_ema2 = batchnorm(conv2_out, tst, iter_step, b_conv2, convolutional=True)
            active3_out = tf.nn.relu(conv3_out)
        with tf.name_scope('Pool2d_3'):
            pool3_out = Pool2d(active3_out)

        with tf.name_scope('Con2d_4'):
            W_conv4 = WeightsVariable([3, 3, 256, 512], name_str='weights_conv4', stddev=0.01)
            b_conv4 = BiasesVariable(shape=[512], name_str='biases_conv4')
            conv4_out = Conv2d(pool3_out, W_conv4, b_conv4, stride=1)
            # Y2bn, update_ema2 = batchnorm(conv2_out, tst, iter_step, b_conv2, convolutional=True)
            active4_out = tf.nn.relu(conv4_out)
        with tf.name_scope('Pool2d_4'):
            pool4_out = Pool2d(active4_out)

        ''' Fully Connect layer '''
        with tf.name_scope('FC_1'):
            fc_input_flat = tf.reshape(pool4_out, [-1, 7 * 7 * 512])
            W_fc1 = WeightsVariable([7 * 7 * 512, 4096], name_str='weights_fc1', stddev=0.00001)
            b_fc1 = BiasesVariable([4096], name_str='biases_fc1')
            fc1_out = FullyConnected(fc_input_flat, W_fc1, b_fc1)
            fc1_out = tf.nn.dropout(fc1_out, 0.8)
        # with tf.name_scope('FC_2'):
        #     W_fc2 = WeightsVariable([4096, 4096])
        #     b_fc2 = BiasesVariable([4096])
        #     # fc2_out_logits = tf.nn.softmax(tf.matmul(fc1_out, W_fc2) + b_fc2)
        #     fc2_out = tf.matmul(fc1_out, W_fc2) + b_fc2

            # tf.add_to_collection('prediction', fc2_out_logits)
            # tf.add_to_collection('X_origin', X_origin)
            # update_ema = tf.group(update_ema1, update_ema2)

        with tf.name_scope('FC_2'):
            W_fc3 = WeightsVariable([4096, FLAGS.n_class])
            b_fc3 = BiasesVariable([FLAGS.n_class])
            # fc2_out_logits = tf.nn.softmax(tf.matmul(fc1_out, W_fc2) + b_fc2)
            fc3_out_logits = tf.matmul(fc1_out, W_fc3) + b_fc3
            tf.add_to_collection('prediction', fc3_out_logits)
            tf.add_to_collection('X_origin', X_origin)
    with tf.name_scope('Loss'):
        # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=fc3_out_logits))
        binary_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=fc3_out_logits))
    ''' train layer '''
    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.015)
        train_op = optimizer.minimize(binary_entropy_loss)

    ''' evaluate_layer'''
    with tf.name_scope('Evaluate'):
        correct_predic = tf.equal((Y_true > 0), (fc3_out_logits > 0))
        accuracy = tf.reduce_mean(tf.cast(correct_predic, tf.float32))

    init = tf.global_variables_initializer()
    summary_writer = tf.summary.FileWriter(logdir='./log', graph=tf.get_default_graph())
    summary_writer.close()


    # mnist_datasets = input_data.read_data_sets('./fashion', one_hot=True)
    # read file name
    file_list = os.listdir(file_dir)
    np.random.shuffle(file_list)
    file_list_length = len(file_list)


    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        sess.run(init)
        total_batches = int(file_list_length / batch_size)
        print("Per batch size: ", batch_size)
        print("Number of samples: ", file_list_length)
        print("Total batches Count: ", total_batches)
        print("Learning Rate:", FLAGS.learning_rate)
        training_step = 0

        for epoch in range(FLAGS.training_epochs):
            for batch_idx in range(total_batches):
                file_train_batch = file_list[batch_size * batch_idx:batch_size * (batch_idx + 1)]

                batch_x, batch_y = utils.get_data(file_train_batch)                                # get batch of training
                # batch_x, batch_y = mnist_datasets.train.next_batch(batch_size)
                # predict_c = sess.run(correct_predic, feed_dict={X_origin:batch_x, Y_true:batch_y, learning_rate:FLAGS.learning_rate})
                sess.run(train_op, feed_dict={X_origin:batch_x, Y_true:batch_y, learning_rate:FLAGS.learning_rate})
                training_step += 1

                ''' calculate the loss of latest display_step '''
                if training_step % FLAGS.display_step == 0:
                    start_idx = max(0, (batch_idx - FLAGS.display_step) * batch_size)
                    end_idx = batch_idx * batch_size
                    # train_loss, train_acc = EvaluateModelOnDataset(sess, mnist_datasets.train.images[start_idx:end_idx, :],
                    #                                                mnist_datasets.train.labels[start_idx:end_idx, :])
                    train_loss, train_acc = EvaluateModelOnDataset(sess, batch_x, batch_y)
                    print("Training Step: " + str(training_step) +
                          ", Training Loss=%.5f" % train_loss +
                          ", Training Accuracy=%.5f" % train_acc)

                    file_val_batch = file_list[batch_size * total_batches:]                             # get batch of validation
                    validation_x, validation_y = utils.get_data(file_val_batch)
                    validation_loss,  validation_acc = EvaluateModelOnDataset(sess, validation_x, validation_y)
                    print("Training Step: " + str(training_step) +
                          ", Validation Loss=%.5f" % validation_loss +
                          ", Validation Accuracy=%.5f" % validation_acc)
            if epoch % 10 == 0:
                print("-------------Save the model-------------")
                saver.save(sess, "./checkpoint_adam/", global_step=epoch)
        print("Done")

        # test_samples_count = mnist_datasets.test.num_examples
        # test_loss, test_accuracy = EvaluateModelOnDataset(sess, mnist_datasets.test.images, mnist_datasets.test.labels)
        # print("Testing Samples Count:", test_samples_count)
        # print("Testing Loss:", test_loss)
        # print("Testing Accuracy:", test_accuracy)

#/usr/bin/python

# multilayer perceptron neural network with softmax layer to classify genetic data
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import sys, argparse
import os

import matplotlib.pyplot as plt

import tf_util


class PointNet:
    def __init__(self, lr=0.001, epochs=75, \
        batch_size=16, disp_step=1, n_points=25, n_input=3, \
        n_classes=4, dropout=0, load=0, save=0, verbose=0):

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = disp_step
        self.n_points = n_points
        self.n_input = n_input
        self.n_classes = n_classes
        self.load = load
        self.save = save
        self.dropout = dropout
        self.verbose = verbose

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # model definition for pointnet
    def pointnet(self, point_cloud, is_training, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        end_points = {}
        input_image = tf.expand_dims(point_cloud, -1)
        
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        
        # MLP on global point cloud vector
        net = tf.layers.flatten(net)
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, self.n_classes, activation_fn=None, scope='fc3')

        return net, end_points


    # get the loss from predictions vs labels
    def get_loss(self, pred, label, end_points):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
        tf.summary.scalar('classify loss', classify_loss)
        return classify_loss


    # method to run the training/evaluation of the model
    def run(self, dataset):

        tf.reset_default_graph()

        pc_pl = tf.placeholder(tf.float32, [None, self.n_points, self.n_input])
        y_pl = tf.placeholder(tf.float32, [None, self.n_classes])
        is_training_pl = tf.placeholder(tf.bool, shape=())  

        # Construct model
        pred, end_points = self.pointnet(pc_pl, is_training_pl)

        loss = self.get_loss(pred, y_pl, end_points)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        
        saver = tf.train.Saver()

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        if self.load:
            saver.restore(sess, '/tmp/cnn')

        total_batch = int(dataset.train.num_examples/self.batch_size)

        is_training = True

        # Training cycle
        for epoch in range(self.epochs):
            avg_cost = 0.
            
            dataset.shuffle()
            
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.train.next_batch(self.batch_size, i)

                #batch_x = dataset.train.permute(batch_x, idxs)
                _, c = sess.run([optimizer, loss], feed_dict={pc_pl: batch_x, 
                                                              y_pl: batch_y,
                                                              is_training_pl: is_training})

                # Compute average loss
                avg_cost += c / total_batch

            if self.verbose:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        if self.save:
            saver.save(sess, "/tmp/cnn")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_pl, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accs = []
        is_training = False
        total_test_batch = int(dataset.test.num_examples / 8192)
        for i in range(total_test_batch):
            batch_x, batch_y = dataset.test.next_batch(self.batch_size, i)
            #batch_x = dataset.train.permute(batch_x, idxs)
            accs.append(accuracy.eval({pc_pl: batch_x, 
                                       y_pl: batch_y,
                                       is_training_pl: is_training}, 
                                       session=sess))

        sess.close()

        print accs

        return sum(accs) / float(len(accs))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_cifar.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf
import threading
import socket
import json

sys.path.append('../')
import loader as loader
from src.nets.vgg import VGG_CIFAR10
from src.helper.trainer import Trainer
from src.helper.evaluator import Evaluator


# DATA_PATH = '/home/junda/cifar-10-batches-py'
# SAVE_PATH = '/home/junda/vgg19/'
scheduler_op = None
target_gpus = []

def binary_to_dict(the_binary):
    jsn = the_binary.decode('utf-8')
    d = json.loads(jsn)
    return d


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--port', type=int,
                        help='Listen port for scheduling message')

    parser.add_argument('--data_path', type=str,
                        help='Specify data location')

    parser.add_argument('--save_path', type=str,
                        help='Specify saved model location')

    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--load', type=int, default=104,
                        help='Epoch id of pre-trained model')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probability for dropout')
    parser.add_argument('--maxepoch', type=int, default=100,
                        help='Max number of epochs for training')

    return parser.parse_args()


def train():
    FLAGS = get_args()
    train_data, valid_data = loader.load_cifar(
        cifar_path=FLAGS.data_path, batch_size=FLAGS.bsize, substract_mean=True)

    train_model = VGG_CIFAR10(
        n_channel=3, n_class=10, pre_trained_path=None,
        bn=True, wd=5e-3, trainable=True, sub_vgg_mean=False)
    train_model.create_train_model()

    valid_model = VGG_CIFAR10(
        n_channel=3, n_class=10, bn=True, sub_vgg_mean=False)
    valid_model.create_test_model()

    trainer = Trainer(train_model, valid_model, train_data, init_lr=FLAGS.lr)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        writer = tf.compat.v1.summary.FileWriter(FLAGS.save_path)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(FLAGS.maxepoch):
            trainer.train_epoch(sess, keep_prob=FLAGS.keep_prob, summary_writer=writer)
            trainer.valid_epoch(sess, dataflow=valid_data, summary_writer=writer)
            if scheduler_op == 'g' or 's':
                saved_model = '{}vgg-cifar-epoch-{}'.format(FLAGS.save_path, epoch_id)
                saver.save(sess, saved_model)
                # generate command
                config_command = '--train --port ' + str(FLAGS.port) + ' --data_path '
                # send command back to the scheduler
                pass
            # saver.save(sess, '{}vgg-cifar-epoch-{}'.format(SAVE_PATH, epoch_id))
        saver.save(sess, '{}vgg-cifar-epoch-{}'.format(FLAGS.save_path, epoch_id))


def evaluate():
    FLAGS = get_args()
    train_data, valid_data = loader.load_cifar(
        cifar_path=FLAGS.data_path, batch_size=FLAGS.bsize, substract_mean=True)

    valid_model = VGG_CIFAR10(
        n_channel=3, n_class=10, bn=True, sub_vgg_mean=False)
    valid_model.create_test_model()

    evaluator = Evaluator(valid_model)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}vgg-cifar-epoch-{}'.format(FLAGS.save_path, FLAGS.load))
        print('training set:', end='')
        evaluator.accuracy(sess, train_data)
        print('testing set:', end='')
        evaluator.accuracy(sess, valid_data)


def msg_handle(port):
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.bind(('0.0.0.0', port))
    connection.listen(10)
    while True:
        current_connection, address = connection.accept()
        data = current_connection.recv(2048)
        msg = binary_to_dict(data)
        if msg['op'] == 'sg' or 'ss':
            # stop previous grow/shrink command
            scheduler_op = 'stop'
        elif msg['op'] == 'g':
            scheduler_op = 'g'
        elif msg['op'] == 's':
            scheduler_op = 's'
        target_gpus = msg['gpus']


if __name__ == "__main__":
    FLAGS = get_args()
    # build message pipeline
    msg_handler = threading.Thread(target=msg_handle, args=(FLAGS.port,))
    if FLAGS.train:
        train()
    if FLAGS.eval:
        evaluate()

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import socket
import json
import numpy as np
import time

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.resnet import resnet_model
from official.resnet import resnet_run_loop
from official.resnet import analyzer

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

DATASET_NAME = 'CIFAR-10'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def dict_to_binary(the_dict):
  message = json.dumps(the_dict)
  return message.encode()


def parse_record(raw_record, is_training):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_IMAGES['train'],
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
  lr = params['learning_rate']
  te = int(params['train_epochs'])
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=params['batch_size'],
      num_images=_NUM_IMAGES['train'], boundary_epochs=[te, te, te],
      decay_rates=[lr, lr, lr, lr])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Cifar10Model,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype'],
      fine_tune=params['fine_tune']
  )


def define_cifar_flags():
  resnet_run_loop.define_resnet_flags()
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(data_dir='/tmp/cifar10_data',
                          model_dir='/tmp/cifar10_model',
                          resnet_size='32',
                          train_epochs=250,
                          epochs_between_evals=10,
                          batch_size=128)


def run_cifar(flags_obj):
  """Run ResNet CIFAR-10 training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                    or input_fn)
  resnet_run_loop.resnet_main(
      flags_obj, cifar10_model_fn, input_function, DATASET_NAME,
      shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])

  # After training, compute training loss&validation accuracy, and return them to the Monitor
  # First get event file
  train_args = []
  eval_args = []
  train_file = ''
  eval_file = ''
  train_dir = flags_obj.md

  for r, d, f in os.walk(train_dir):
    for file in f:
      if 'tfevents' in file:
        if 'eval' in r:
          eval_file = r + '/' + file
        else:
          train_file = r + '/' + file

  if flags_obj.status == 'init':
    # collect and plot, then connect server
    # show model_directory files and find events file
    train_args = analyzer.training_args(train_file, int(flags_obj.bs))
    #TODO uptrend define to fill abnormal
    abnormal = 0
    '''
    for r, d, f in os.walk(eval_dir):
      for file in f:
        if 'tfevents' in file:
          # This is the training events file
          eval_args = analyzer.validation_args(file)
          break
    '''

    if len(train_args) > 0: # and len(eval_args) > 0:
      # connect server and return the data
      sendback(flags_obj, train_args, eval_args, abnormal)

  elif flags_obj.status == 'train':
    # just save the data/evaluate
    # TODO evaluate model performance
    # train_args = analyzer.training_args(train_file, int(flags_obj.bs))
    train_args = []
    abnormal = 0
    sendback(flags_obj, train_args, eval_args, abnormal)
    pass


def sendback(flags_obj, train_args, eval_args, abnormal):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  ip_address, port = flags_obj.server.split(':')
  server_address = (ip_address, int(port))
  sock.connect(server_address)
  try:
    message = {}
    message['status'] = flags_obj.status
    message['gpu_info'] = (flags_obj.ni, flags_obj.gi)
    message['loc'] = flags_obj.md
    # temporarily no need of eval_args for prediction model
    message['init_f'] = [float(flags_obj.lr), np.log2(flags_obj.bs)] + list(train_args)
    # complete time
    message['cpt_tm'] = time.time()
    message['ab'] = abnormal
    message['exec_tm'] = flags_obj.et
    message['wait_tm'] = float(flags_obj.wt)
    message['id'] = flags_obj.jid
    message['te'] = flags_obj.te
    sock.sendall(dict_to_binary(message))
  finally:
    sock.close()



def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_cifar(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_cifar_flags()
  absl_app.run(main)

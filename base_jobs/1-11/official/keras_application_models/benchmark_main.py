# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark on the keras built-in application models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

# information transfer
import threading
import socket
import json
import sys
import os
import time
sys.path.insert(0, '../../')

from official.keras_application_models import dataset
from official.keras_application_models import model_callbacks
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils

# Define a dictionary that maps model names to their model classes inside Keras
MODELS = {
        "vgg16": tf.keras.applications.VGG16,
        "vgg19": tf.keras.applications.VGG19,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "xception": tf.keras.applications.Xception,
        "resnet50": tf.keras.applications.ResNet50,
        "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
        "mobilenet": tf.keras.applications.MobileNet,
        "densenet121": tf.keras.applications.DenseNet121,
        "densenet169": tf.keras.applications.DenseNet169,
        "densenet201": tf.keras.applications.DenseNet201,
        "nasnetlarge": tf.keras.applications.NASNetLarge,
        "nasnetmobile": tf.keras.applications.NASNetMobile,
}

job_status = 'normal'
node = ''
gpus = []
exit_code = False
training_flags = 0
have_trained = 0
lock = True


def binary_to_dict(the_binary):
    jsn = the_binary.decode('utf-8')
    d = json.loads(jsn)
    return d


def receive(server_ip, port):
    # open a server and wait for job report
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        connection.bind((server_ip, port))
    except:
        # original job does not close the port
        # kill original job process
        os.system('kill $(lsof -t -i:' + str(port) + ')')
        time.sleep(2)
        connection.bind((server_ip, port))

    connection.listen(10)
    while not exit_code:
        current_connection, address = connection.accept()
        data = current_connection.recv(2048)
        info = binary_to_dict(data)
        global job_status, node, gpus
        job_status = info['status']
        node = info['node']
        gpus = info['gpus']
    print('Connection closed')
    connection.close()


def dict_to_binary(the_dict):
    message = json.dumps(the_dict)
    return message.encode()


def send_msg(address, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip_address, port = address.split(':')
    server_address = (ip_address, int(port))
    sock.connect(server_address)
    try:
        sock.sendall(dict_to_binary(message))
    except:
        print(message)
    finally:
        sock.close()


def run_keras_model_benchmark(_):
    new_job_thread = threading.Thread(target=receive, 
        args=(FLAGS.server_address.split(':')[0], FLAGS.port,), daemon=True)
    new_job_thread.start()
    """Run the benchmark on keras model."""
    # Ensure a valid model name was supplied via command line argument
    if FLAGS.model not in MODELS.keys():
        raise AssertionError("The --model command line argument should "
                                                 "be a key in the `MODELS` dictionary.")
    # print(FLAGS.gpus_list)
    # exit()
    # Check if eager execution is enabled
    if FLAGS.eager:
        tf.logging.info("Eager execution is enabled...")
        tf.enable_eager_execution()

    # Load the model
    tf.logging.info("Benchmark on {} model...".format(FLAGS.model))
    keras_model = MODELS[FLAGS.model]
    model = keras_model(weights=None)

    # Get dataset
    dataset_name = "ImageNet"
    if FLAGS.use_synthetic_data:
        tf.logging.info("Using synthetic dataset...")
        dataset_name += "_Synthetic"
        train_dataset = dataset.generate_synthetic_input_dataset(
                FLAGS.model, FLAGS.batch_size)
        val_dataset = dataset.generate_synthetic_input_dataset(
                FLAGS.model, FLAGS.batch_size)
    else:
        raise ValueError("Only synthetic dataset is supported!")

    num_gpus = flags_core.get_num_gpus(FLAGS)

    distribution = None
    # Use distribution strategy
    if FLAGS.dist_strat:
        distribution = distribution_utils.get_distribution_strategy(
                num_gpus=num_gpus)
    elif num_gpus > 1:
        # Run with multi_gpu_model
        # If eager execution is enabled, only one GPU is utilized even if multiple
        # GPUs are provided.
        if FLAGS.eager:
            tf.logging.warning(
                    "{} GPUs are provided, but only one GPU is utilized as "
                    "eager execution is enabled.".format(num_gpus))
        model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus)

    # Adam optimizer and some other optimizers doesn't work well with
    # distribution strategy (b/113076709)
    # Use GradientDescentOptimizer here
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy",
                                optimizer=optimizer,
                                metrics=["accuracy"],
                                distribute=distribution)

    # Create benchmark logger for benchmark logging
    run_params = {
            "batch_size": FLAGS.batch_size,
            "synthetic_data": FLAGS.use_synthetic_data,
            "train_epochs": FLAGS.train_epochs,
            "num_train_images": FLAGS.num_train_images,
            "num_eval_images": FLAGS.num_eval_images,
    }

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info(
            model_name=FLAGS.model,
            dataset_name=dataset_name,
            run_params=run_params,
            test_id=FLAGS.benchmark_test_id)

    class LossHistory(tf.keras.callbacks.Callback):
        def __init__(self):
            self.start = time.time()

        def on_train_begin(self, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            global training_flags, have_trained
            if job_status == 'g':
                training_flags = 1
                have_trained = epoch + 1
                self.model.stop_training = True
            if job_status == 's':
                training_flags = 1
                have_trained = epoch + 1
                self.model.stop_training = True

        def on_batch_end(self, batch, logs={}):
            global lock
            if batch == 49 and lock is True:
                hundred = time.time() - self.start
                # calculate the speed and unlock job
                msg = {}
                msg['id'] = FLAGS.id
                msg['status'] = 'un'
                msg['ep_tm'] = FLAGS.num_train_images * hundred / (FLAGS.batch_size * 50)
                send_msg(FLAGS.server_address, msg)
                lock = False

    # Create callbacks that log metric values about the training and evaluation
    callbacks = model_callbacks.get_model_callbacks(
            FLAGS.callbacks,
            batch_size=FLAGS.batch_size,
            metric_logger=benchmark_logger)
    callbacks.append(LossHistory())
    # Train and evaluate the model
    history = model.fit(
            train_dataset,
            epochs=FLAGS.train_epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            steps_per_epoch=int(np.ceil(FLAGS.num_train_images / FLAGS.batch_size)),
    )

    ''' No need for evaluation part
    tf.logging.info("Logging the evaluation results...")
    for epoch in range(FLAGS.train_epochs):
        eval_results = {
                "accuracy": history.history["val_acc"][epoch],
                "loss": history.history["val_loss"][epoch],
                tf.GraphKeys.GLOBAL_STEP: (epoch + 1) * np.ceil(
                        FLAGS.num_eval_images/FLAGS.batch_size)
        }
        benchmark_logger.log_evaluation_result(eval_results)
    '''

    # Clear the session explicitly to avoid session delete error
    tf.keras.backend.clear_session()
    # Now end the training send back message
    msg = {}
    remain_ep = FLAGS.train_epochs - have_trained
    if training_flags == 0 or remain_ep == 0:
        msg['status'] = 'e'
        msg['id'] = FLAGS.id
        # send_msg(FLAGS.server_address, msg)
    else:
        # ask the scheduler to re-run
        # growing is needed
        gpus_loc = {}
        flags_gpu_list = [int(i) for i in FLAGS.gpus_list]
        if job_status == 'g':
            new_gpus_list = gpus + flags_gpu_list
            msg['status'] = 'g'
        else:
            new_gpus_list = list(set(flags_gpu_list).difference(set(gpus)))
            msg['status'] = 's'
        # TODO hardcoded here
        gpus_loc['localhost'] = new_gpus_list
        msg['gpus_loc'] = gpus_loc
        msg['id'] = FLAGS.id
        msg['ep'] = FLAGS.train_epochs - have_trained
        # send_msg(FLAGS.server_address, msg)

    global exit_code
    exit_code = True
    time.sleep(1)
    send_msg(FLAGS.server_address, msg)
    print('exit')
    exit()


def define_keras_benchmark_flags():
    """Add flags for keras built-in application models."""
    flags_core.define_base(hooks=False)
    flags_core.define_performance()
    flags_core.define_image()
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags_core.set_defaults(
            data_format="channels_last",
            use_synthetic_data=True,
            batch_size=32,
            train_epochs=2)

    flags.DEFINE_enum(
            name="model", default=None,
            enum_values=MODELS.keys(), case_sensitive=False,
            help=flags_core.help_wrap(
                    "Model to be benchmarked."))

    flags.DEFINE_integer(
            name="num_train_images", default=1000,
            help=flags_core.help_wrap(
                    "The number of synthetic images for training. The default value is "
                    "1000."))

    flags.DEFINE_integer(
            name="num_eval_images", default=50,
            help=flags_core.help_wrap(
                    "The number of synthetic images for evaluation. The default value is "
                    "50."))

    flags.DEFINE_boolean(
            name="eager", default=False, help=flags_core.help_wrap(
                    "To enable eager execution. Note that if eager execution is enabled, "
                    "only one GPU is utilized even if multiple GPUs are provided and "
                    "multi_gpu_model is used."))

    flags.DEFINE_boolean(
            name="dist_strat", default=False, help=flags_core.help_wrap(
                    "To enable distribution strategy for model training and evaluation. "
                    "Number of GPUs used for distribution strategy can be set by the "
                    "argument --num_gpus."))

    flags.DEFINE_list(
            name="callbacks",
            default=["ExamplesPerSecondCallback", "LoggingMetricCallback"],
            help=flags_core.help_wrap(
                    "A list of (case insensitive) strings to specify the names of "
                    "callbacks. For example: `--callbacks ExamplesPerSecondCallback,"
                    "LoggingMetricCallback`"))

    # below is for scheduler
    flags.DEFINE_integer(
        name="port", default=8888,
        help=flags_core.help_wrap(
            "The port that receive message from the scheduler"))

    flags.DEFINE_list(
        name="gpus_list",
        default=[],
        help=flags_core.help_wrap(
            'GPUs that this job will occupy'))

    # TODO combine node and gpu info into a flag
    flags.DEFINE_string(
        name="node",
        default='',
        help=flags_core.help_wrap(
            'Which Node to run'))

    flags.DEFINE_string(
        name="server_address",
        default='',
        help=flags_core.help_wrap(
            'Monitor server address'))

    flags.DEFINE_string(
        name="id",
        default='',
        help=flags_core.help_wrap(
            'Monitor server address'))

    @flags.multi_flags_validator(
            ["eager", "dist_strat"],
            message="Both --eager and --dist_strat were set. Only one can be "
                            "defined, as DistributionStrategy is not supported in Eager "
                            "execution currently.")
    # pylint: disable=unused-variable
    def _check_eager_dist_strat(flag_dict):
        return not(flag_dict["eager"] and flag_dict["dist_strat"])


def main(_):
    with logger.benchmark_context(FLAGS):
        run_keras_model_benchmark(FLAGS)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_keras_benchmark_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)

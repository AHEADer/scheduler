import time
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import json
import os
from datetime import datetime

t = time.time()
'''
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["localhost:2222"]
    },
   "task": {"type": "worker", "index": 0}
})
'''

Num_class = 1000
Input_size = 224
bs = 32
step = 1
num_gpus = 1

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_config))


def input_fn(batch_size):
    input_shape = [Input_size, Input_size, 3]
    input_element = nest.map_structure(lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape))
    label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([1]))
    element = (input_element, label_element)
    ds = tf.data.Dataset.from_tensors(element).repeat().batch(batch_size)
    return ds


def dummy():
    pass


class _LoggerHook(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""
    def begin(self):
        print('train start')
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        # return tf.train.SessionRunArgs(loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        if self._step % 1 == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = 1 * 32 / duration
            sec_per_batch = float(duration)

            format_str = '%s: step %d, (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), self._step, examples_per_sec, sec_per_batch))


def model_main():
    model = tf.keras.applications.ResNet50(weights=None)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    train_distribute = tf.contrib.distribute.MirroredStrategy(
        num_gpus_per_worker=num_gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    config = tf.estimator.RunConfig()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier = tf.keras.estimator.model_to_estimator(keras_model=model, config=config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(batch_size=bs), max_steps=step)# , hooks=[_LoggerHook()])
    eval_spec=tf.estimator.EvalSpec(input_fn=lambda: dummy())
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


model_main()
ts = time.time()-t
print(ts)

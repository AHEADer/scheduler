import tensorflow as tf
import matplotlib.pyplot as plt
import loss_fit
import scipy.optimize as optimize
import scipy.stats as stats


def train_data(train_file):
    steps = []
    values = []
    for e in tf.train.summary_iterator(train_file):
        for v in e.summary.value:
            if v.tag == 'cross_entropy_1':
                steps.append(e.step)
                values.append(v.simple_value)
    return steps, values


def validate_data(validate_file):
    steps = []
    values = []
    for e in tf.train.summary_iterator(validate_file):
        for v in e.summary.value:
            if v.tag == 'accuracy':
                steps.append(e.step)
                values.append(v.simple_value)
    length = len(steps)
    epochs = [i+1 for i in range(length)]
    return epochs, values


def inv_func(x, a, b, c, d):
    return c/(a*x+b) + d


def training_args(train_file):
    x, y = train_data(train_file)
    a, b, c, d = optimize.curve_fit(inv_func, x, y)[0]
    return a, b, c, d


def validation_args(validate_file):
    x, y = validate_data(validate_file)
    y_min = min(y)
    f = lambda t, k, theta, conv: (conv - y_min) * stats.gamma.cdf(t, a=k, scale=theta) + y_min
    k, theta, conv = optimize.curve_fit(f, x, y)[0]
    return k, theta, conv

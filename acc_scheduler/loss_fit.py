import numpy as np
import scipy.optimize as optimize
import csv
import scipy.stats as stats
import plot
import matplotlib.pyplot as plt


def get_values(data_file, batch_size, epoch):
    count = 0
    X = []
    Y = []
    current_max = 0
    # flag=0: original, flag=1, keep maximum, flag=2 avg
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                count += 1
                if float(row[1])*batch_size/50000 > epoch:
                    break
                X.append(float(row[1])*batch_size/50000)
                Y.append(float(row[2]))
        return X[1:], Y[1:]


def inv_func(x, a, b, c, d):
    return c/(a*x+b) + d


def exp_func(x, a, b, c):
    return a*np.exp(-b*x, dtype=np.float64)+c


def get_inv_args(file_name, bs, epoch):
    x, y = get_values(file_name, bs, epoch)
    plt.plot(x, y)
    a, b, c, d = optimize.curve_fit(inv_func, x, y)[0]
    inv_y = [inv_func(i, a, b, c, d) for i in x]
    plt.plot(x, inv_y)
    return a, b, c, d


def get_exp_args(file_name, bs, epoch):
    x, y = get_values(file_name, bs, epoch)
    a, b, c = optimize.curve_fit(exp_func, x, y)[0]
    exp_y = [exp_func(i, a, b, c) for i in x]
    plt.plot(x, exp_y)
    plt.show()
    return a, b, c


def get_gamma_arg(acc_file_name, epoch_n):
    pre_x, pre_y = plot.get_values(acc_file_name, 0)
    x = pre_x[:epoch_n]
    y = pre_y[:epoch_n]
    y_min = min(y)
    f = lambda x, k, theta, conv: (conv - y_min) * stats.gamma.cdf(x, a=k, scale=theta) + y_min
    k, theta, conv = optimize.curve_fit(f, x, y)[0]
    new_y = [f(i, k, theta, conv) for i in x]
    plt.plot(x, y)
    plt.plot(x, new_y)
    plt.show()
    return k, theta, conv
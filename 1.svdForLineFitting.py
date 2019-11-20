# coding=utf-8
import numpy as np
import random
from matplotlib import pyplot as plt
import math


def default_function(x):
    return 2 * x + 4


def getXData(start=0, end=100, interval=1):
    """
    generate x data with no noise

    :param start:
    :param end:
    :param interval:
    :return:
    """
    x_data = []
    times = int((end - start) / interval)
    for i in range(times):
        value = start + i * interval
        x_data.append(value)
    return x_data


def getYData(x_data, model=default_function):
    """
    calculate y data using input model with no noise

    :param x_data:
    :param model:
    :return:
    """
    y_data = []
    for i in range(len(x_data)):
        y_data.append(model(x_data[i]))
    return y_data


def addNoise(data_series, mu=0, sigma=1):
    """
    add a gaussian-distributed noise to data sequence

    :param data_series:
    :param mu:
    :param sigma:
    :return:
    """

    noise_data = []
    for i in range(len(data_series)):
        noise_data.append(data_series[i] + random.gauss(mu, sigma))
    return noise_data


def joinXY(x_data, y_data):
    """
    join x and y in to one

    :param x_data:
    :param y_data:
    :return:
    """
    points = []
    for i in range(len(x_data)):
        points.append([x_data[i], y_data[i]])
    return points


def lsp_svd(x_data, y_data):
    """
    solving least square problem with SVD

    :param x_data:
    :param y_data:
    :return:
    """

    num = len(x_data)
    A = np.ones([num, 3], np.float)
    for i in range(len(x_data)):
        A[i, 0] = x_data[i]
        A[i, 1] = y_data[i]
    u, s, vt = np.linalg.svd(A)
    v = np.transpose(vt)
    solution = v[:, -1]
    return solution


if __name__ == '__main__':
    for i in range(10):
        # generate ideal data(hidden state)
        x_data = getXData(1, 100)
        y_data = getYData(x_data)

        # generate measurement data with noise(measurement)
        x_data_noise = addNoise(x_data, sigma=1)
        y_data_noise = addNoise(y_data, sigma=1)

        # estimate line equation using date with noise
        [a, b, c] = lsp_svd(x_data_noise, y_data_noise)

        # calculate estimate error
        y_data_est = []
        errors = []
        for i in range(len(x_data_noise)):
            error = pow(a * x_data_noise[i] + b * y_data_noise[i] + c, 2)
            errors.append(error)
            y_est = (-a / b) * x_data_noise[i] + (-c / b)
            y_data_est.append(y_est)
        mean_error = math.sqrt(np.sum(errors))

        print str(round(a, 3)) + "x+" + str(round(b, 3)) + "y+" + str(round(c, 3)) + "=0"
        print "y=" + str(round(-a / b, 3)) + "x+" + str(round(-c / b, 3))
        print "mean error:", round(mean_error, 3), "\n"

        plt.title("Estimate line equation using SVD")
        plt.grid()
        plt.tight_layout()
        plt.plot(x_data, y_data, label='original function', color='red')
        plt.plot(x_data_noise, y_data_est, label='estimate function', color='orange')
        plt.scatter(x_data_noise, y_data_noise, label='noised data')
        plt.legend()
        plt.show()

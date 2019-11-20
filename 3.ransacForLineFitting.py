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

    print "=>Generating data from", start, "to", end, "with interval", interval
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

    print "=>Calculating y data according to given model and x data"
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

    print "=>Adding gaussian-distributed noise to data"
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


def randomSampling(data_series, n=2):
    select_items = []
    for i in range(n):
        select_items.append(data_series[random.randint(0, len(data_series) - 1)])
    return select_items


def cvtSamplePoints2XY(samples):
    point_x = []
    point_y = []
    for i in range(len(samples)):
        point_x.append(samples[i][0])
        point_y.append(samples[i][1])
    return point_x, point_y


def addOutliers(data_series, num_outlier, mu=0, sigma=10):
    """
    add some outliers to data

    :param data_series:
    :param mu:
    :param sigma:
    :return:
    """

    print "=>Adding outliers to data"
    noise_data = data_series[:]
    for i in range(num_outlier):
        index = random.randint(0, len(noise_data) - 1)
        noise_data[index] = noise_data[index] + random.gauss(mu, sigma)
    return noise_data


if __name__ == '__main__':
    # input parameters
    start_x = 1
    x_interval = 1
    end_x = 100
    noise_sigma = 1
    num_outliers = 30
    outlier_sigma = 15
    ep = 7
    w = 70.0 / 100.0
    n = 2.0
    p = 0.995

    # generate data
    x_data = getXData(start_x, end_x, interval=x_interval)
    y_data = getYData(x_data)

    x_data_noise = addNoise(x_data, sigma=noise_sigma)
    y_data_noise = addNoise(y_data, sigma=noise_sigma)

    # add some outliers to data
    x_data_noise_outlier = addOutliers(x_data_noise, num_outliers, sigma=outlier_sigma)
    y_data_noise_outlier = addOutliers(y_data_noise, num_outliers, sigma=outlier_sigma)

    print "=>Showing generated data\n"
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.title("data with noise")
    plt.scatter(x_data_noise, y_data_noise)
    plt.subplot(1, 2, 2)
    plt.tight_layout()
    plt.title("data with noise and outliers")
    plt.scatter(x_data_noise_outlier, y_data_noise_outlier)
    plt.show()

    point_data_noise_outlier = joinXY(x_data_noise_outlier, y_data_noise_outlier)

    good_coefs = []
    good_nums = []
    errors = []
    inliers_x = []
    inliers_y = []

    k = int(math.log(1 - p) / math.log(1 - math.pow(w, n))) + 1
    print "=>To estimate correct model with " + round(p * 100, 2).__str__() + \
          "% certainty,you should iterate for", k, "times at least"

    for it in range(k):
        # random sampling
        samples = randomSampling(point_data_noise_outlier, 2)
        print "\nTime", it + 1, "/", k
        print "selected points:\n", samples[0], samples[1]
        points_x, points_y = cvtSamplePoints2XY(samples)

        # model using random samples
        a, b, c = lsp_svd(points_x, points_y)  # ax+by+c=0
        k2 = -a / b
        b2 = -c / b
        print "estimate function:\n", str(round(a, 3)) + "x+" + str(round(b, 3)) + "y+" + str(round(c, 3)) + "=0"
        print "y=" + str(round(k2, 3)) + "x+" + str(round(b2, 3))

        # calculate offset function
        theta = math.atan(k2)
        b_ = ep / math.cos(theta)
        y_data_up = []
        y_data_low = []
        for i in range(len(x_data)):
            y_data_up.append(k2 * x_data[i] + (b2 + b_))
            y_data_low.append(k2 * x_data[i] + (b2 - b_))

        # thresholding and inlier counting
        tmp_error = 0
        counter = 0
        good_pts_x = []
        good_pts_y = []
        y_data_est = []
        for i in range(len(x_data_noise_outlier)):
            y_ = (-a / b) * x_data_noise_outlier[i] - (c / b)
            dy = abs(y_data_noise_outlier[i] - y_)
            k_ = -a / b
            theta = math.atan(k_)
            error = dy * math.cos(theta)

            tmp_error += error
            if error <= ep:
                counter += 1
                good_pts_x.append(x_data_noise_outlier[i])
                good_pts_y.append(y_data_noise_outlier[i])
            y_est = (-a / b) * x_data_noise_outlier[i] + (-c / b)
            y_data_est.append(y_est)

        mean_error = tmp_error / len(x_data_noise_outlier)
        errors.append(mean_error)
        good_coefs.append([a, b, c])
        good_nums.append(counter)
        print "good point number:\n", counter, "/", len(x_data_noise)
        print "mean error:\n", mean_error, "\n"

        if counter > len(inliers_x):
            inliers_x = []
            inliers_y = []
            for i in range(len(good_pts_x)):
                inliers_x.append(good_pts_x[i])
                inliers_y.append(good_pts_y[i])

        plt.title("Time " + (it + 1).__str__() + "/" + k.__str__() + " inliers=" + counter.__str__().zfill(
            2) + " mean error=" + round(mean_error, 2).__str__())
        plt.tight_layout()
        plt.plot(x_data_noise_outlier, y_data_est, label='estimate function', color='orange')
        plt.plot(x_data, y_data_up, label='up range', color='green')
        plt.plot(x_data, y_data_low, label='low range', color='green')

        plt.scatter(x_data_noise_outlier, y_data_noise_outlier, label='noised data')
        plt.scatter(good_pts_x, good_pts_y, label='consensus points', color='red')
        plt.scatter(points_x, points_y, label='selected points', color='yellow')
        plt.legend()
        plt.show()

        # clear variables
        tmp_error = 0
        counter = 0
        good_pts_x = []
        good_pts_y = []
        y_data_est = []

    best = good_nums.index(max(good_nums))
    print "======Result======"
    print "best consensus point number:\n", max(good_nums), "/", len(x_data_noise)
    print "best coefficients(a,b,c):\n", good_coefs[best]
    print "estimate function:\n", str(round(good_coefs[best][0], 3)) + "x+" + str(
        round(good_coefs[best][1], 3)) + "y+" + str(round(good_coefs[best][2], 3)) + "=0"
    print "y=" + str(round(-good_coefs[best][0] / good_coefs[best][1], 3)) + "x+" + str(
        round(-good_coefs[best][2] / good_coefs[best][1], 3))
    print "inliers:"
    for i in range(len(inliers_x)):
        print "inlier", (i + 1).__str__().zfill(3), "(", inliers_x[i], ",", inliers_y[i], ")"


    plt.title("Selected " + len(inliers_x).__str__() + " inliers")
    plt.scatter(x_data_noise_outlier, y_data_noise_outlier, label="outliers")
    plt.scatter(inliers_x, inliers_y, label="inliers")
    plt.tight_layout()
    plt.legend()
    plt.show()

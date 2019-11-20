# coding=utf-8
import numpy as np
import random
from matplotlib import pyplot as plt
import math


def getPointPairData(x_start=0, y_start=0, x_end=60, y_end=60, point_num=100, offset_x=35, offset_y=40):
    print "=>Generating random-distributed point data"
    x_data1 = []
    y_data1 = []
    x_data2 = []
    y_data2 = []
    for i in range(point_num):
        x_data1.append(random.randint(x_start, x_end))
        y_data1.append(random.randint(y_start, y_end))
        x_data2.append(x_data1[i] + offset_x)
        y_data2.append(y_data1[i] + offset_y)
    return x_data1, y_data1, x_data2, y_data2


def addNoiseToPointPair(x_data1, y_data1, x_data2, y_data2, mu=0, sigma=1):
    print "=>Adding gaussian-distributed noise to data"
    noise_x_data1 = []
    noise_y_data1 = []
    noise_x_data2 = []
    noise_y_data2 = []
    for i in range(len(x_data1)):
        noise_x_data1.append(x_data1[i] + random.gauss(mu, sigma))
        noise_y_data1.append(y_data1[i] + random.gauss(mu, sigma))
        noise_x_data2.append(x_data2[i] + random.gauss(mu, sigma))
        noise_y_data2.append(y_data2[i] + random.gauss(mu, sigma))
    return noise_x_data1, noise_y_data1, noise_x_data2, noise_y_data2


def drawMatchPairs(x_data1, y_data1, x_data2, y_data2):
    plt.title("Point pairs")
    plt.scatter(x_data1, y_data1, label="original point")
    plt.scatter(x_data2, y_data2, label="matched point")
    for i in range(len(x_data1)):
        plt.plot([x_data1[i], x_data2[i]], [y_data1[i], y_data2[i]], color="skyblue")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()


def lsp_svd(x_data1, y_data1, x_data2, y_data2):
    print "=>Calculating Homography matrix"
    A = []
    for i in range(len(x_data1)):
        ax = [-x_data1[i], -y_data1[i], -1, 0, 0, 0, x_data1[i] * x_data2[i], y_data1[i] * x_data2[i], x_data2[i]]
        ay = [0, 0, 0, -x_data1[i], -y_data1[i], -1, x_data1[i] * y_data2[i], y_data1[i] * y_data2[i],
              y_data2[i]]
        A.append(ax)
        A.append(ay)

    A = np.array(A)
    u, s, vt = np.linalg.svd(A)
    v = np.transpose(vt)
    solution = v[:, -1]
    h = np.zeros([3, 3], np.float)
    h[0, 0] = solution[0]
    h[0, 1] = solution[1]
    h[0, 2] = solution[2]
    h[1, 0] = solution[3]
    h[1, 1] = solution[4]
    h[1, 2] = solution[5]
    h[2, 0] = solution[6]
    h[2, 1] = solution[7]
    h[2, 2] = solution[8]
    return h


def checkAccuracy(x_data1, y_data1, x_data2, y_data2, homo):
    print "=>Checking homography transformation accuracy on every point"
    error_x = []
    error_y = []
    error = []
    for i in range(len(x_data1)):
        p1 = np.ones([3], np.float)
        p1[0] = x_data1[i]
        p1[1] = y_data1[i]

        p2 = np.matmul(homo, p1)
        p2_normal = p2 / p2[2]

        reprojection_error_x = abs(x_data2[i] - p2_normal[0])
        reprojection_error_y = abs(y_data2[i] - p2_normal[1])
        reprojection_error = math.sqrt(math.pow(reprojection_error_x, 2) + math.pow(reprojection_error_y, 2))
        error_x.append(reprojection_error_x)
        error_y.append(reprojection_error_y)
        error.append(reprojection_error)

    max_ex = np.max(error_x)
    min_ex = np.min(error_x)
    mean_ex = np.mean(error_x)
    max_ey = np.max(error_y)
    min_ey = np.min(error_y)
    mean_ey = np.mean(error_y)
    max_e = np.max(error)
    min_e = np.min(error)
    mean_e = np.mean(error)
    print "\t\tmax\t\tmin\t\tmean"
    print "x\t\t", round(max_ex, 3), "\t", round(min_ex, 3), "\t", round(mean_ex, 3)
    print "y\t\t", round(max_ey, 3), "\t", round(min_ey, 3), "\t", round(mean_ey, 3)
    print "total\t", round(max_e, 3), "\t", round(min_e, 3), "\t", round(mean_e, 3)
    return error_x, error_y, error


if __name__ == '__main__':
    x_data1, y_data1, x_data2, y_data2 = getPointPairData(point_num=50)
    x_data1_n, y_data1_n, x_data2_n, y_data2_n = addNoiseToPointPair(x_data1, y_data1, x_data2, y_data2)
    h = lsp_svd(x_data1_n, y_data1_n, x_data2_n, y_data2_n)
    print "Homography matrix:\n", h

    checkAccuracy(x_data1_n, y_data1_n, x_data2_n, y_data2_n, h)

    drawMatchPairs(x_data1_n, y_data1_n, x_data2_n, y_data2_n)

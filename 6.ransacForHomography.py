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


def addOutliersToPointPair(x_data1, y_data1, x_data2, y_data2, num_outlier, mu=0, sigma=15):
    print "=>Adding outliers to data"
    noise_x_data1 = x_data1[:]
    noise_y_data1 = y_data1[:]
    noise_x_data2 = x_data2[:]
    noise_y_data2 = y_data2[:]
    indices = []
    for i in range(num_outlier):
        index = random.randint(0, len(x_data1) - 1)
        indices.append(index)
        noise_x_data1[index] = x_data1[index] + random.gauss(mu, sigma)
        noise_y_data1[index] = y_data1[index] + random.gauss(mu, sigma)
        noise_x_data2[index] = x_data2[index] + random.gauss(mu, sigma)
        noise_y_data2[index] = x_data2[index] + random.gauss(mu, sigma)
    return noise_x_data1, noise_y_data1, noise_x_data2, noise_y_data2, indices


def drawMatchPairsWithOutliers(x_data1, y_data1, x_data2, y_data2, indices, color="lightgreen"):
    plt.title("Point pairs")
    plt.scatter(x_data1, y_data1, label="original point")
    plt.scatter(x_data2, y_data2, label="matched point")
    for i in range(len(x_data1)):
        plt.plot([x_data1[i], x_data2[i]], [y_data1[i], y_data2[i]], color=color)
    for i in range(len(indices)):
        plt.plot([x_data1[indices[i]], x_data2[indices[i]]], [y_data1[indices[i]], y_data2[indices[i]]], color="red")
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
    return error_x, error_y, error


def randomSampling(x_data1, y_data1, x_data2, y_data2, n=4):
    x_data1_s = []
    y_data1_s = []
    x_data2_s = []
    y_data2_s = []
    for i in range(int(n)):
        tmp_index = random.randint(0, len(x_data1) - 1)
        x_data1_s.append(x_data1[tmp_index])
        y_data1_s.append(y_data1[tmp_index])
        x_data2_s.append(x_data2[tmp_index])
        y_data2_s.append(y_data2[tmp_index])
    return x_data1_s, y_data1_s, x_data2_s, y_data2_s


if __name__ == '__main__':
    start_x = 0
    end_x = 60
    start_y = 0
    end_y = 60
    num_points = 20
    offset_x = 35
    offset_y = 40
    num_outliers = 3
    noise_sigma = 0.5
    outlier_sigma = 2
    ep = 5
    w = 1.0 * (num_points - num_outliers) / num_points
    n = 4.0
    p = 0.995

    x_data1, y_data1, x_data2, y_data2 = getPointPairData(start_x, start_y,
                                                          end_x, end_y,
                                                          num_points,
                                                          offset_x, offset_y)
    x_data1_n, y_data1_n, x_data2_n, y_data2_n = addNoiseToPointPair(x_data1, y_data1, x_data2, y_data2,
                                                                     sigma=noise_sigma)
    x_data1_o, y_data1_o, x_data2_o, y_data2_o, indices = addOutliersToPointPair(x_data1_n, y_data1_n,
                                                                                 x_data2_n, y_data2_n,
                                                                                 num_outlier=num_outliers,
                                                                                 sigma=outlier_sigma)
    print "=>Showing generated data\n"
    drawMatchPairsWithOutliers(x_data1_o, y_data1_o, x_data2_o, y_data2_o, indices)

    good_coefs = []
    good_nums = []
    errors = []
    inliers_pts = []
    k = int(math.log(1 - p) / math.log(1 - math.pow(w, n))) + 1
    print "=>To estimate correct model with " + round(p * 100, 2).__str__() + \
          "% certainty,you should iterate for", k, "times at least"

    for it in range(k):
        sample_x1, sample_y1, sample_x2, sample_y2 = randomSampling(x_data1_o, y_data1_o, x_data2_o, y_data2_o, n=n)
        print "\nTime", it + 1, "/", k
        print "selected point pairs:"
        for j in range(len(sample_x1)):
            print sample_x1[j], sample_y1[j], "->", sample_x2[j], sample_y2[j]

        h = lsp_svd(sample_x1, sample_y1, sample_x2, sample_y2)
        print "Homography matrix:\n", h

        tmp_error = 0
        counter = 0
        good_pts = []
        pts_est_x = []
        pts_est_y = []
        for i in range(len(x_data1_o)):
            p1 = np.ones([3], np.float)
            p1[0] = x_data1_o[i]
            p1[1] = y_data1_o[i]

            p2 = np.matmul(h, p1)
            p2_normal = p2 / p2[2]
            pts_est_x.append(p2_normal[0])
            pts_est_y.append(p2_normal[1])

            reprojection_error_x = abs(x_data2_o[i] - p2_normal[0])
            reprojection_error_y = abs(y_data2_o[i] - p2_normal[1])
            reprojection_error = math.sqrt(math.pow(reprojection_error_x, 2) + math.pow(reprojection_error_y, 2))
            tmp_error += reprojection_error
            if reprojection_error <= ep:
                counter += 1
                good_pts.append(i)
        mean_error = tmp_error / len(x_data1_o)
        errors.append(mean_error)
        good_coefs.append(h)
        good_nums.append(counter)
        print "good point number:\n", counter, "/", len(x_data1_o)
        print "mean error:\n", mean_error, "\n"

        if counter > len(inliers_pts):
            inliers_pts = []
            for i in range(len(good_pts)):
                inliers_pts.append(good_pts[i])

        plt.scatter(x_data1_o, y_data1_o, label="original point")
        plt.scatter(x_data2_o, y_data2_o, label="matched point")
        plt.scatter(pts_est_x, pts_est_y, label='estimate point')
        for i in range(len(x_data1_o)):
            if i == 0:
                plt.plot([x_data2_o[i], pts_est_x[i]], [y_data2_o[i], pts_est_y[i]], color="pink",
                         label="estimate error")
                plt.plot([x_data1_o[i], x_data2_o[i]], [y_data1_o[i], y_data2_o[i]], color="lightgreen",
                         label="unselected pair")
            else:
                plt.plot([x_data2_o[i], pts_est_x[i]], [y_data2_o[i], pts_est_y[i]], color="pink")
                plt.plot([x_data1_o[i], x_data2_o[i]], [y_data1_o[i], y_data2_o[i]], color="lightgreen")
        for i in range(len(good_pts)):
            if i == 0:
                plt.plot([x_data1_o[good_pts[i]], x_data2_o[good_pts[i]]],
                         [y_data1_o[good_pts[i]], y_data2_o[good_pts[i]]],
                         color="red", label="selected pair")
            else:
                plt.plot([x_data1_o[good_pts[i]], x_data2_o[good_pts[i]]],
                         [y_data1_o[good_pts[i]], y_data2_o[good_pts[i]]],
                         color="red")
        plt.title("Time " + (it + 1).__str__() + "/" + k.__str__() + " inliers=" + counter.__str__().zfill(2))
        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.show()

        tmp_error = 0
        counter = 0
        good_pts = []
        pts_est = []

    best = good_nums.index(max(good_nums))
    print "======Result======"
    print "best consensus point number:\n", max(good_nums), "/", len(x_data1_o)
    print "best homography:\n", good_coefs[best]
    print "inliers:"
    for i in range(len(inliers_pts)):
        print "inlier", (i + 1).__str__().zfill(3), \
            x_data1_o[inliers_pts[i]], \
            y_data1_o[inliers_pts[i]], "->", \
            x_data2_o[inliers_pts[i]], \
            y_data2_o[inliers_pts[i]]

    plt.scatter(x_data1_o, y_data1_o, label="original data")
    plt.scatter(x_data2_o, y_data2_o, label="match data")
    for i in range(len(x_data1_o)):
        if i == 0:
            plt.plot([x_data1_o[i], x_data2_o[i]], [y_data1_o[i], y_data2_o[i]], color="lightgreen", label="outliers")
        else:
            plt.plot([x_data1_o[i], x_data2_o[i]], [y_data1_o[i], y_data2_o[i]], color="lightgreen")
    for i in range(len(inliers_pts)):
        if i == 0:
            plt.plot([x_data1_o[inliers_pts[i]], x_data2_o[inliers_pts[i]]],
                     [y_data1_o[inliers_pts[i]], y_data2_o[inliers_pts[i]]], color="red", label="inliers")
        else:
            plt.plot([x_data1_o[inliers_pts[i]], x_data2_o[inliers_pts[i]]],
                     [y_data1_o[inliers_pts[i]], y_data2_o[inliers_pts[i]]], color="red")

    save_data = open("save_homo.txt", "w")
    save_data.write("original data:\n")
    for i in range(len(x_data1_o)):
        save_data.write(
            x_data1_o[i].__str__() + "\t" + y_data1_o[i].__str__() + "\t" +
            x_data2_o[i].__str__() + "\t" + y_data2_o[i].__str__() + "\n")
    save_data.write("estimated homography:\n")
    save_data.write(h.__str__() + "\n")
    save_data.write("inliers:\n")
    for i in range(len(inliers_pts)):
        save_data.write(x_data1_o[inliers_pts[i]].__str__() + "\t" + x_data2_o[inliers_pts[i]].__str__() + "\t" +
                        y_data1_o[inliers_pts[i]].__str__() + "\t" + y_data2_o[inliers_pts[i]].__str__() + "\n")
    save_data.close()

    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()

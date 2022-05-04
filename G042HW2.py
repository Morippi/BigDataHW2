import time
import sys
import math
import random


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


def euclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return math.sqrt(res)


def minDistance(P, n):
    subset = P[:n]
    min_d = 99999
    for i in subset:
        for j in subset:
            current_d = euclidean(i, subset[j])
            if current_d < min_d:
                min_d = current_d
    return min_d // 2


def pointsInRadius(Z, x, r):
    # returns set of points in radius r from x
    ball = []
    for i in Z:
        if euclidean(i, x) < r:
            ball.append(i)
    return ball


def SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0):
    # r <- (Min distance between first k+z+1 points) / 2
    r = minDistance(inputPoints, k + z + 1)
    Z = inputPoints
    S = []
    W_z = sum(weights)
    while len(S) < k and W_z > 0:
        new_center = tuple()
        MAX = 0
        for x in inputPoints:
            # ball weight is the sum of weights of all point in the radius (1+2*alpha)r
            W_y = pointsInRadius(Z, x, (1 + 2 * alpha) * r)
            ball_weight = sum(W_y)
            if ball_weight > MAX:
                MAX = ball_weight
                new_center = x
        S.append(new_center)
    return S


if __name__ == '__main__':
    assert len(sys.argv) == 4, "Usage: python G042Hw2.py <file_name> <k> <z>"

    file_name = sys.argv[1]
    inputPoints = readVectorsSeq(file_name)
    weights = [1 for i in range(len(inputPoints))]

    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    z = sys.argv[3]
    assert z.isdigit(), "z must be an integer"
    z = int(z)

    SeqWeightedOutliers(inputPoints, weights, k, z)

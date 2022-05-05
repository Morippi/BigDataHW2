import sys
import math


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
    index = 0
    # i = subset.pop(0)
    while subset:
        i = subset.pop()
        for j in subset:
            current_d = euclidean(i, j)
            # print("d between ", i, " and ", j, ": ", current_d)
            if current_d < min_d:
                min_d = current_d

    # print(min_d)
    return min_d / 2


def pointsInRadius(P, w, x, r):
    # returns set of points in radius r from x
    ball = []
    # print("x: ", x)
    for i in range(len(P)):
        # print(P[i])
        if euclidean(P[i], x) < r:
            # print(euclidean(P[i], x), " is in radius")
            ball.append(w[i])
    return ball


def SeqWeightedOutliers(inputPoints, weights, k, z, alpha=1):
    # r <- (Min distance between first k+z+1 points) / 2
    r = minDistance(inputPoints, k + z + 1)
    num_iter = 0
    print("radius: ", r)
    S = []
    Z = inputPoints
    W_z = sum(weights)
    while len(S) < k:
        new_center = tuple()
        MAX = 0
        for x in Z:
            # ball weight is the sum of weights of all point in the radius (1+2*alpha)r
            W_y = pointsInRadius(Z, weights, x, (1 + 2 * alpha) * r)
            ball_weight = sum(W_y)
            if ball_weight > MAX:
                MAX = ball_weight
                new_center = x
        S.append(new_center)

        for y in Z:
            if euclidean(y, new_center) < (3 + 4 * alpha) * r:
                W_z -= weights[Z.index(y)]
                Z.remove(y)

        if W_z <= z:
            return S
        else:
            r = 2 * r
            num_iter += 1

    print("S: ", S)
    return S, r, num_iter


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

    r_init = minDistance(inputPoints, k + z + 1)
    S, r_fin, num_iter = SeqWeightedOutliers(inputPoints, weights, k, z)

    # OUTPUT
    print("Input size n =  ", len(inputPoints))
    print("Number of centers k =  ", k)
    print("Number of ouliers z =  ", z)
    print("Initial guess =  ", r_init)
    print("Final guess =  ", r_fin)
    print("Number of guesses =  ", num_iter)
    print("Objective function =  ")
    print("Time of SeqWeightedOutliers = ")

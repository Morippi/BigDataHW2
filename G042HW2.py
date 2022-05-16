import sys
import math
import copy
from time import perf_counter


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


def minDistance(P, EuclidianDistance, n):
    subset = P[:n]
    min_d = math.inf
    while subset:
        i = subset.pop()
        for j in subset:

            current_d = EuclidianDistance[i][j]

            # print("d between ", i, " and ", j, ": ", current_d)
            if current_d < min_d:
                min_d = current_d
    # print(min_d)
    return min_d / 2


def pointsInRadius(P, EuclidianDistance, w, x, r):
    # returns set of points in radius r from x
    ball_weight = 0
    # print("x: ", x)
    for point in P:
        # print(P[i])
        if EuclidianDistance[x][point] < r:
            # print(euclidean(P[i], x), " is in radius")
            ball_weight += P[point]
    return ball_weight


def SeqWeightedOutliers(inputPoints, EuclidianDistance, weights, k, z, alpha=0):
    # r <- (Min distance between first k+z+1 points) / 2
    r = minDistance(inputPoints, EuclidianDistance, k + z + 1)

    num_iter = 1
    while True:
        Z = {}
        for i in range(len(inputPoints)):
            Z[inputPoints[i]] = weights[i]
        S = []
        W_z = sum(weights)

        op = (1 + 2 * alpha) * r
        while len(S) < k and W_z > 0:
            MAX = 0
            new_center = tuple()

            for x in Z:
                # ball weight is the sum of weights of all point in the radius (1+2*alpha)r

                ball_weight = pointsInRadius(Z, EuclidianDistance, weights, x, op)
                if ball_weight > MAX:
                    MAX = ball_weight
                    new_center = x
            S.append(new_center)
            points_to_remove = []

            for y in Z:

                if EuclidianDistance[y][new_center] < (3 + 4 * alpha) * r:
                    W_z -= Z[y]
                    points_to_remove.append(y)

            for point_to_remove in points_to_remove:
                del Z[point_to_remove]
        if W_z <= z:
            break
        else:
            r = 2 * r
            num_iter += 1
    return S, r, num_iter


def PrecompileDistance(inputPoints):
    dict = {}
    for i in data:
        dict[i] = {}
        dict[i][i] = 0
    while (inputPoints):
        i = inputPoints.pop()

        for j in inputPoints:
            d = euclidean(i, j)
            dict[i][j] = d
            dict[j][i] = d

    return dict


def ComputeObjective(P, S, z):
    # At first we compute for each point the closest center.
    # for each point we save:
    # - the distance to the closest center
    # - the center
    distances = []
    for point in P:
        min_distance = float('inf')
        closest_center = None
        for center in S:
            distance = euclidean(point, center)
            if min_distance > distance:
                min_distance = distance
                closest_center = center
        distances.append((min_distance, closest_center))

    # We sort the list on the distances
    distances = sorted(distances, key=lambda couple: couple[0])

    # We pop the list z times to remove the top z distances
    for i in range(z):
        distances.pop()

    # We save for each center the maximum distance
    max_distances = {}
    for center in S:
        max_distances[center] = 0

    for distance, center in distances:
        max_distances[center] = max(max_distances[center], distance)

    # The sum of the maximum distances between the center of a cluster and the point
    # in the same cluster is the objective value
    objective_value = 0
    for center in max_distances:
        objective_value = max(objective_value, max_distances[center])

    return objective_value


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

    data = inputPoints[:]

    EuclidianDistance = PrecompileDistance(data)
    r_init = minDistance(inputPoints, EuclidianDistance, k + z + 1)

    start_time = perf_counter()

    S, r_fin, num_iter = SeqWeightedOutliers(inputPoints, EuclidianDistance, weights, k, z)
    end_time = perf_counter()
    time = int((end_time - start_time) * 1000)

    objective = ComputeObjective(inputPoints, S, z)
    # OUTPUT
    print("Input size n = ", len(inputPoints))
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Initial guess = ", r_init)
    print("Final guess = ", r_fin)
    print("Number of guesses = ", num_iter)
    print("Objective function = ", objective)
    print("Time of SeqWeightedOutliers = ", time)

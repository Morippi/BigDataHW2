import sys
import math
import copy
from time import perf_counter
import numpy as np


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
    min_d = math.inf
    while subset:
        i = subset.pop()
        for j in subset:

            current_d = euclidean(i, j)

            # print("d between ", i, " and ", j, ": ", current_d)
            if current_d < min_d:
                min_d = current_d
    # print(min_d)
    return min_d / 2


def pointsInRadius(point_array, weight_array, x, x_w, first_circle_squared):
    squared_distances = np.sum(np.square(point_array - x), 1)
    indeces = np.where(squared_distances < first_circle_squared)
    return weight_array[indeces].sum() - x_w


def SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0):
    # r <- (Min distance between first k+z+1 points) / 2
    r = minDistance(inputPoints, k + z + 1)

    num_iter = 1
    while True:
        Z_points = np.zeros((len(inputPoints), len(inputPoints[0])))
        Z_weight = np.zeros(len(inputPoints))
        for index in range(len(inputPoints)):
            Z_points[index] = np.asarray(inputPoints[index])
            Z_weight[index] = weights[index]
        S = []
        W_z = np.sum(weights)

        op = (1 + 2 * alpha) * r
        while len(S) < k and W_z > 0:
            first_circle_squared = r ** 2
            MAX = -1
            new_center = None
            for index in range(len(Z_points)):
                # ball weight is the sum of weights of all point in the radius (1+2*alpha)r
                x = Z_points[index]
                x_w = Z_weight[index]
                ball_weight = pointsInRadius(Z_points, Z_weight, x, x_w, first_circle_squared)
                if ball_weight > MAX:
                    MAX = ball_weight
                    new_center = x

            S.append(tuple(new_center))

            points_to_maintain = []
            second_circle_squared = ((3 + 4 * alpha) * r) ** 2
            for indeces in range(len(Z_points)):
                if np.sum(np.square(Z_points[indeces] - new_center)) < second_circle_squared:
                    W_z -= Z_weight[indeces]
                else:
                    points_to_maintain.append(indeces)

            Z_points = Z_points[points_to_maintain]
            Z_weight = Z_weight[points_to_maintain]
        if W_z <= z:
            break
        else:
            r = 2 * r
            num_iter += 1
    return S, r, num_iter


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

    r_init = minDistance(inputPoints, k + z + 1)

    start_time = perf_counter()

    S, r_fin, num_iter = SeqWeightedOutliers(inputPoints, weights, k, z)
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

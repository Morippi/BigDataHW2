import sys
import math
import numpy as np
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


def minDistance(P, n):

    min_d = 99999
    for i in P:
        for j in P:
            current_d = np.linalg.norm(i-j)
            # print("d between ", i, " and ", j, ": ", current_d)
            if current_d < min_d and current_d:
                min_d = current_d
    # print(min_d)
    return min_d / 2


def pointsInRadius(P, w, x, r):
    # returns set of points in radius r from x
    ball_weight = 0
    # print("x: ", x)
    for point in range(len(P)):
        # print(P[i])
        if np.linalg.norm(P[point]-x) < r:
            # print(euclidean(P[i], x), " is in radius")
            ball_weight += w[point]
    return ball_weight


def SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0):
    # r <- (Min distance between first k+z+1 points) / 2
    r = minDistance(inputPoints, k + z + 1)

    num_iter = 1
    while True:


        S = []
        W_z = sum(weights)
        while len(S) < k and W_z > 0:
            MAX = 0



            for x in inputPoints:

                # ball weight is the sum of weights of all point in the radius (1+2*alpha)r
                ball_weight = pointsInRadius(inputPoints, weights, x, (1 + 2 * alpha) * r)
                if ball_weight > MAX:
                    MAX = ball_weight
                    new_center = x
            S.append(new_center)


            points_to_remove = []
            for y in range(len(inputPoints)):
                if np.linalg.norm(y - new_center) < (3 + 4 * alpha) * r:
                    W_z -= weights[y]
                    points_to_remove.append(y)



            for point_to_remove in range(len(points_to_remove)):
                np.delete(inputPoints, point_to_remove, 0)

        if W_z <= z:
            break
        else:
            r = 2 * r
            num_iter += 1
        print(num_iter, W_z, z)

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
        objective_value = max(objective_value,max_distances[center])

    return objective_value

if __name__ == '__main__':
    assert len(sys.argv) == 4, "Usage: python G042Hw2.py <file_name> <k> <z>"

    file_name = sys.argv[1]
    inputPoints = np.loadtxt(file_name, dtype=float, delimiter=',')
    print(inputPoints)
    weights = [1 for i in range(len(inputPoints))]

    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    z = sys.argv[3]
    assert z.isdigit(), "z must be an integer"
    z = int(z)

    r_init = minDistance(inputPoints, k + z + 1)
    start_time = perf_counter()
    S, r_fin, num_iter = SeqWeightedOutliers(inputPoints, weights, k, z)
    end_time = perf_counter()
    time = int((end_time - start_time)*1000)

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

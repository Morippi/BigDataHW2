import sys
from time import perf_counter
import math
import numpy as np

#Fuction that opens the file and put the data in a list of tuple
def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result

#Fuction of the euclidian distance between two points
def euclidean(point1, point2):
    dist = [(a - b) ** 2 for a, b in zip(point1, point2)]
    return math.sqrt(sum(dist))

#PART 1
#Given a list of point P and a integer n,
#return the minimum distance between the first n point of P
def minDistance(P, n):
    subset = np.array(P[:n])
    min_d = float('inf')
    while len(subset):
        i = subset[0]
        subset=np.delete(subset,0,0)
        for j in subset:
            #current_d = euclidean(i, j)
            current_d = np.sqrt(np.sum(np.square(i - j)))

            if  current_d < min_d:
                min_d = current_d
    # print(min_d)
    return min_d / 2

#given an numpu array of point, a numpy array of the weight of the points
#a point x (with weight x_w) and a radious op
#return the weight of the points that are inside the sphere
#with center x and radious op.
def weightInRadius(point_array, weight_array, x, x_w, op):
    # we used the np to make this step more efficient
    # we firstly compute the euclidean distances from x to all the point in point_array
    euclidean_distance = np.sqrt(np.sum(np.square(point_array - x), 1))

    # we find the indeces of the point that are inside the first ball
    indeces = np.where(euclidean_distance < op)

    #we get the weight of those points and we remove
    #the weight of x
    return weight_array[indeces].sum() - x_w


def SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0):
    #we compute the first circle base radious
    r = minDistance(inputPoints, k + z + 1)
    r_init = r;
    num_iter = 1
    while True:
        # To make the this function more efficient we are going to use Numpy arrays
        # to reppresent Z.
        # Z_weight is going to store the weights of the point in Z, using same indexing
        Z = np.zeros((len(inputPoints), len(inputPoints[0])))
        Z_weight = np.zeros(len(inputPoints))
        for index in range(len(inputPoints)):
            Z[index] = np.asarray(inputPoints[index])
            Z_weight[index] = weights[index]

        # We initialize the set of the centers solutions
        S = []
        # We compute the initial Weight of Z
        W_z = np.sum(Z_weight)

        op = (1 + 2 * alpha) * r
        while len(S) < k and W_z > 0:
            #we initialize max distance and new_center
            MAX = -1
            new_center = None

            #for each point we compute the weight inside the relative ball
            for index in range(len(Z)):

                x = Z[index]
                x_w = Z_weight[index]

                ball_weight = weightInRadius(Z, Z_weight, x, x_w, op)
                if ball_weight > MAX:
                    MAX = ball_weight
                    new_center = x

            #we add the new center to the solutions
            S.append(tuple(new_center))

            #we collect the index of the point outside the sedond ball of the new centers
            points_to_maintain = []
            for indeces in range(len(Z)):
                if np.sqrt(np.sum(np.square(Z[indeces] - new_center))) <= ((3 + 4 * alpha) * r):
                    W_z -= Z_weight[indeces]
                else:
                    points_to_maintain.append(indeces)

            # remove points that are not in the bigger circle from Z;
            Z = Z[points_to_maintain]
            Z_weight = Z_weight[points_to_maintain]

        if W_z <= z:
            break
        else:
            r = 2 * r
            num_iter += 1

    return S, r_init, r, num_iter

#PART 2
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

#PART 3
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

    start_time = perf_counter()

    S, r_init, r_fin, num_iter = SeqWeightedOutliers(inputPoints, weights, k, z)
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

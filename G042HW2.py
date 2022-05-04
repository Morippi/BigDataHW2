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

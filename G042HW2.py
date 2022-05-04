import time
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


if __name__ == '__main__':

    assert len(sys.argv) == 2, "path to dataset:"
    file_name = sys.argv[1]
    dataset = readVectorsSeq(file_name)
    #print(dataset)
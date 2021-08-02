import numpy as np
import math
import gc


class Histogram:

    def __init__(self, _dimSize=16, _range=256):
        self.dimSize = _dimSize
        self.range = _range
        self.data = []
        self.rangePerBinInv = 1.0 / (_range / _dimSize)
        self.initialized = False

    def insertValues(self, data1, data2, data3, weight):
        self.data = np.zeros(self.dimSize ** 3, dtype=np.float64).tolist()
        if len(data1) != len(weight):
            _weight = np.ones(len(data1), dtype=np.float64).tolist()
        else:
            _weight = weight

        for index in range(len(data1)):
            id1 = int(self.rangePerBinInv * data1[index])
            id2 = int(self.rangePerBinInv * data2[index])
            id3 = int(self.rangePerBinInv * data3[index])
            id = id1 * self.dimSize * self.dimSize + id2 * self.dimSize + id3
            self.data[id] += _weight[int(index)]
        self.normalize()

    def computeSimilarity(self, hist):
        conf = 0.
        for index in range(len(self.data)):
            conf += math.sqrt(self.data[index] * hist.data[index])
        return conf

    def getValue(self, val):
        id1 = int(self.rangePerBinInv * val[0])
        id2 = int(self.rangePerBinInv * val[1])
        id3 = int(self.rangePerBinInv * val[2])
        id = id1 * self.dimSize * self.dimSize + id2 * self.dimSize + id3
        return self.data[id]

    def transformToWeights(self):
        _min = 1.
        for num in self.data:
            if num < _min and num != 0:
                _min = num
        self.transformByWeight(min=_min)

    def transformByWeight(self, min):
        for index, num in enumerate(self.data):
            if num > 0:
                self.data[index] = 1.0 * min / num
                if self.data[index] > 1:
                    self.data[index] = 1
            else:
                self.data[index] = 1

    def multiplyByWeights(self, hist):
        assert len(self.data) == len(hist.data)
        for index, num in enumerate(self.data):
            self.data[index] = num * hist.data[index]
        self.normalize()

    def clear(self):
        self.data = []
        gc.collect()

    def normalize(self):
        self.data = self.data / np.sum(self.data)

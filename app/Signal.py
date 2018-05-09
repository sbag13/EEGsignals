import numpy as np
from Epoch import Epoch
import time

class Signal():
    def __init__(self, label, samples, frequency, attr=None):
        # can add personal info
        self.label = label
        self.data = np.asarray(samples)
        self.frequency = frequency
        if attr is None:
            self.attr = {}
        else:
            self.attr = attr
    def getEpochs(self):
        epochs = []
        samples_in_epoch = 30 * self.frequency
        i = 0
        while i < self.data.shape[0]:
            j = i + samples_in_epoch
            epochs.append(Epoch(self.data[i:j]))
            i = j
        return epochs

def createInput(epochs, verbose=False):
    inputMatrix = []
    size = len(epochs)
    current = 0
    start = time.time()
    for e in epochs:
        current += 1
        if(verbose==True):
            print("%d / %d" % (current , size))
        e.extractFeatures()
        inputMatrix.append(list(e.features.values()))
    stop = time.time()
    print("features extracted in: %f sec" % (stop - start))
    return np.array(inputMatrix)
import numpy as np
from Epoch import Epoch

class Signal():
    def __init__(self, label, samples, frequency, attr=None):
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
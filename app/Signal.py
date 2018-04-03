from numpy import asarray
from pywt import wavedec

class Signal():
    def __init__(self, label, samples, frequency, attr=None):
        # can add personal info
        self.label = label
        self.data = asarray(samples)
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

class Epoch():
    def __init__(self, data):
        self.data = asarray(data)
    
    def extractFeatures(self):
        self.coeffs = wavedec(self.data, 'db1, level=7)
        for a in self.coeffs:
            print(len(a))

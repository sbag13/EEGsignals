import numpy as np
import pywt

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

class Epoch():
    def __init__(self, data):
        self.data = np.asarray(data)
    
    def extractFeatures(self):
        self.wp = pywt.WaveletPacket(data=self.data, wavelet='db1', mode='symmetric')
        features = {}
        features['deltaEnergy'] = self.getEnergy(['aaaaaad', 'aaaaad', 'aaaada', 'aaaadd'])
        features['thetaEnergy'] = self.getEnergy(['aaadaa', 'aaadad', 'aaadd', 'aadaa', 'aadada'])
        features['alphaEnergy'] = self.getEnergy(['aadadd', 'aadda'])
        features['spindleEnergy'] = self.getEnergy(['aaddd', 'adaaa', 'adaad'])
        features['beta1Energy'] = self.getEnergy(['adad', 'adda'])
        features['beta2Energy'] = self.getEnergy(['addd', 'da'])
        print(features)
    
    def getEnergy(self, keys):
        return np.mean([self.energy(self.wp[key].data) for key in keys])

    def energy(self, data):
        return np.sqrt(np.sum([x**2 for x in data]) / len(data))
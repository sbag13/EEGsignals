import numpy as np
import pywt
from statistics import stdev 

class Epoch():
    def __init__(self, data):
        self.data = np.asarray(data)
    
    def extractFeatures(self):
        self.wp = pywt.WaveletPacket(data=self.data, wavelet='db1', mode='symmetric')
        self.features = {}

        self.features['deltaEnergy'] = self.getEnergy(['aaaaaad', 'aaaaad', 'aaaada', 'aaaadd'])
        self.features['thetaEnergy'] = self.getEnergy(['aaadaa', 'aaadad', 'aaadd', 'aadaa', 'aadada'])
        self.features['alphaEnergy'] = self.getEnergy(['aadadd', 'aadda'])
        self.features['spindleEnergy'] = self.getEnergy(['aaddd', 'adaaa', 'adaad'])
        self.features['beta1Energy'] = self.getEnergy(['adad', 'adda'])
        self.features['beta2Energy'] = self.getEnergy(['addd', 'da'])
        totalEnergy = sum(self.features.values())
        self.features['totalEnergy'] = totalEnergy
        self.features['alpha2DeltaPlusTheta'] = \
            self.features['alphaEnergy'] / (self.features['deltaEnergy'] + self.features['thetaEnergy'])
        self.features['delta2AlphaPlusTheta'] = \
            self.features['deltaEnergy'] / (self.features['alphaEnergy'] + self.features['thetaEnergy'])
        self.features['theta2DeltaPlusAlpha'] = \
            self.features['thetaEnergy'] / (self.features['deltaEnergy'] + self.features['alphaEnergy'])
        self.allCoeffs = self.getAllCoeffs()
        self.features['allCoefficientsMean'] = np.mean(self.allCoeffs)
        self.features['standardDevaition'] = stdev(self.allCoeffs)

    def getAllCoeffs(self):
        nodes = [self.wp.a.a.a.a.a.a.d, \
            self.wp.a.a.a.a.a.d, \
            self.wp.a.a.a.a.d.a, \
            self.wp.a.a.a.a.d.d, \
            self.wp.a.a.a.d.a.a, \
            self.wp.a.a.a.d.a.d, \
            self.wp.a.a.a.d.d, \
            self.wp.a.a.d.a.a, \
            self.wp.a.a.d.a.d.a, \
            self.wp.a.a.d.a.d.d, \
            self.wp.a.a.d.d.a, \
            self.wp.a.a.d.d.d, \
            self.wp.a.d.a.a.a, \
            self.wp.a.d.a.a.d, \
            self.wp.a.d.a.d, \
            self.wp.a.d.d.a, \
            self.wp.a.d.d.d, \
            self.wp.d.a]
        allCoeffs = []
        for n in nodes:
            allCoeffs.extend(n.data)
        return allCoeffs    

    def getEnergy(self, keys):
        return np.mean([self.energy(self.wp[key].data) for key in keys])

    def energy(self, data):
        return np.sum([x**2 for x in data])
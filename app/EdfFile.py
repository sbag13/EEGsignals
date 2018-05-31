import numpy as np
from Signal import Signal
import pyedflib
import time
import sys

def zero():
    return [1,0,0,0]
def one():
    return [0,1,0,0]
def two():
    return [0,0,1,0]
def three():
    return [0,0,0,1]
def four():
    return [0,0,0,0]

stageList = {0 : zero,
           1 : one,
           2 : two,
           3 : three,
           4 : four
}

class EdfFile():
    def __init__(self, path, annotations_path):
        # check whether the .edf file or not
        self.file = pyedflib.EdfReader(path)
        self.number_of_signal = self.file.signals_in_file
        self.signals_list = []
        for i in np.arange(self.number_of_signal):
            self.signals_list.append(   \
                Signal(self.file.getLabel(i), self.file.readSignal(i),  \
                       self.file.getSampleFrequency(i)))
        self.annotations_file = pyedflib.EdfReader(annotations_path)
        self.annotations = self.annotations_file.readAnnotations()

    def print_labels(self):
        for signal in self.signals_list:
            print(signal.label)

    def createOutput(self, epochs):
        self.stagesMap = {'Sleep stage W' : 0, \
                          'Sleep stage 1' : 1, \
                          'Sleep stage 2' : 2, \
                          'Sleep stage 3' : 3, \
                          'Sleep stage 4' : 3, \
                          'Sleep stage R' : 1, \
                          'Sleep stage ?' : 4, \
                          'Movement time' : 4}
        output = []
        currentEpoch = 0
        currentX = 0
        currentStageIndex = 0
        while currentEpoch < len(epochs):
            if currentStageIndex != len(self.annotations[0]) - 1 :
                if currentX >= self.annotations[0][currentStageIndex + 1]:
                    currentStageIndex = currentStageIndex + 1
            stageNumber = self.stagesMap[self.annotations[2][currentStageIndex]]
            output.append(stageList[stageNumber]())
            currentX = currentX + 30
            currentEpoch = currentEpoch + 1
        return output

    def createInput(self, epochs, verbose=False):
        inputMatrix = []
        size = len(epochs)
        current = 0
        start = time.time()
        for e in epochs:
            current += 1
            if(verbose==True):
                sys.stdout.write("\r%d / %d   " % (current , size))
                sys.stdout.flush()
            e.extractFeatures()
            inputMatrix.append(list(e.features.values()))
        stop = time.time()
        print("\nfeatures extracted in: %f sec" % (stop - start))

        normalized = np.array(inputMatrix).T
        for feature in normalized:
            min = np.min(feature)
            max = np.max(feature)
            for idx in range(len(feature)):
                feature[idx] = (feature[idx] - min)/(max - min) - 0.5

        return normalized.T
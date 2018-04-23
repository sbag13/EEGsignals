import numpy as np
from Signal import Signal
import pyedflib
# import wfdb

def zero():
    return [1,0,0,0,0,0,0]
def one():
    return [0,1,0,0,0,0,0]
def two():
    return [0,0,1,0,0,0,0]
def three():
    return [0,0,0,1,0,0,0]
def four():
    return [0,0,0,0,1,0,0]
def five():
    return [0,0,0,0,0,1,0]
def six():
    return [0,0,0,0,0,0,1]

stageList = {0 : zero,
           1 : one,
           2 : two,
           3 : three,
           4 : four,
           5 : five,
           6 : six,
}

class EdfFile():
    def __init__(self, path):
        # check whether the .edf file or not
        self.file = pyedflib.EdfReader(path)
        self.number_of_signal = self.file.signals_in_file
        self.signals_list = []
        for i in np.arange(self.number_of_signal):
            self.signals_list.append(   \
                Signal(self.file.getLabel(i), self.file.readSignal(i),  \
                       self.file.getSampleFrequency(i)))
        self.annotations_file = pyedflib.EdfReader("../SC4001EC-Hypnogram.edf")     # zrobić uniwersalnie
        self.annotations = self.annotations_file.readAnnotations()
    def print_labels(self):
        for signal in self.signals_list:
            print(signal.label)
    def createOutput(self, epochsCount):
        self.stagesMap = {'Sleep stage W' : 0, \
                          'Sleep stage 1' : 1, \
                          'Sleep stage 2' : 2, \
                          'Sleep stage 3' : 3, \
                          'Sleep stage 4' : 4, \
                          'Sleep stage R' : 5, \
                          'Sleep stage ?' : 6 }
        output = []
        currentEpoch = 0
        currentX = 0
        currentStageIndex = 0
        while currentEpoch < epochsCount:
            if currentX >= self.annotations[0][currentStageIndex + 1]:
                currentStageIndex = currentStageIndex + 1
            stageNumber = self.stagesMap[self.annotations[2][currentStageIndex]]
            output.append(stageList[stageNumber]())
            currentX = currentX + 30
            currentEpoch = currentEpoch + 1
        return output

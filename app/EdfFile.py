from numpy import arange
from Signal import Signal
import pyedflib
# import wfdb

class EdfFile():
    def __init__(self, path):
        # check whether the .edf file or not
        self.file = pyedflib.EdfReader(path)
        self.number_of_signal = self.file.signals_in_file
        self.signals_list = []
        for i in arange(self.number_of_signal):
            self.signals_list.append(   \
                Signal(self.file.getLabel(i), self.file.readSignal(i),  \
                       self.file.getSampleFrequency(i)))
        self.annotations_file = pyedflib.EdfReader("../SC4001EC-Hypnogram.edf")     # zrobić uniwersalnie
        self.annotations = self.annotations_file.readAnnotations()
    def print_labels(self):
        for signal in self.signals_list:
            print(signal.label)
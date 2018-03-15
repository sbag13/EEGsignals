import pyedflib
import numpy

class Signal():
    def __init__(self, label, samples=None, frequency=None, attr=None):
        # can add personal info
        self.label = label
        self.data = numpy.asarray(samples)
        if attr is None:
            self.attr = {}
        else:
            self.attr = attr

class EdfFile():
    def __init__(self, path):
        # check whether the .edf file or not
        self.file = pyedflib.EdfReader(path)
        self.number_of_signal = self.file.signals_in_file
        self.signals_list = []
        for i in numpy.arange(self.number_of_signal):
            self.signals_list.append(   \
                Signal(self.file.getLabel(i), self.file.readSignal(i),  \
                       self.file.getSampleFrequency(i)))
    def print_labels(self):
        for signal in self.signals_list:
            print(signal.label)

edf_file = EdfFile("../SC4001E0-PSG.edf")
edf_file.print_labels()
print(edf_file.signals_list[1].data)

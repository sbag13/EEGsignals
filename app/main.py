import pyedflib
import numpy

edf_file = pyedflib.EdfReader("../SC4001E0-PSG.edf")    # dać wybór
number_of_signals = edf_file.signals_in_file
signals_labels = edf_file.getSignalLabels()

signal_bufs_list = []
for i in numpy.arange(number_of_signals):
    signal_bufs_list.append(edf_file.readSignal(i))
    print(signal_bufs_list[i].shape)

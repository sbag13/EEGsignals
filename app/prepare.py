import numpy as np
import EdfFile
import NN
import Signal

edf_file = EdfFile.EdfFile("../SC4001E0-PSG.edf")
epochs = edf_file.signals_list[1].getEpochs()
outputMatrix = edf_file.createOutput(epochs)

for e, o in zip(epochs, outputMatrix):
    if max(o) == 0:
        epochs.remove(e)
        outputMatrix.remove(o)

inputMatrix = edf_file.createInput(epochs, True)

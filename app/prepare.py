import numpy as np
import EdfFile
import NN

edf_file = EdfFile.EdfFile("../SC4001E0-PSG.edf")
epochs = edf_file.signals_list[1].getEpochs()
inputMatrix = NN.createInput(epochs, True)
output = np.array(edf_file.createOutput(len(epochs)))



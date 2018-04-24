from NN import *
edf_file = EdfFile("../SC4001E0-PSG.edf")
epochs = edf_file.signals_list[1].getEpochs()
inputMatrix = createInputMatrix(epochs, True)
output = np.array(edf_file.createOutput(len(epochs)))
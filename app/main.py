from EdfFile import *
import time


edf_file = EdfFile("../SC4001E0-PSG.edf")
epochs = np.array([edf_file.signals_list[1].getEpochs()])
output = np.array([edf_file.createOutput(epochs.size)]).T
print(epochs.shape)
print(output.shape)
# start = time.time()
# for e in epochs:
#     e.extractFeatures()
# stop = time.time()
# print("features extracted in: %f sec" % (stop - start))
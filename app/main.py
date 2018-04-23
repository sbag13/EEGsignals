from EdfFile import EdfFile
import numpy as np
import time

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

edf_file = EdfFile("../SC4001E0-PSG.edf")
epochs = np.array([edf_file.signals_list[1].getEpochs()])
output = np.array(edf_file.createOutput(epochs.size))

print(epochs.shape)
print(output.shape)

np.random.seed(1)

syn0 = 2*np.random.random((12,8)) - 1
syn1 = 2*np.random.random((8,7)) - 1

# for j in range(output.shape[0]):
#     l0 = 


# start = time.time()
# for e in epochs:
#     e.extractFeatures()
# stop = time.time()
# print("features extracted in: %f sec" % (stop - start))

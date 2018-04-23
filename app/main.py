from EdfFile import EdfFile
import numpy as np
import time

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def createInputMatrix(epochs, verbose=False):
    inputMatrix = []
    size = len(epochs)
    current = 0
    for e in epochs:
        current += 1
        if(verbose==True):
            print("%d / %d" % (current , size))
        e.extractFeatures()
        inputMatrix.append(list(e.features.values()))
    return np.array(inputMatrix)

# todo rozdzielic na dwa skrypty może
edf_file = EdfFile("../SC4001E0-PSG.edf")

epochs = edf_file.signals_list[1].getEpochs()
inputMatrix = createInputMatrix(epochs, True)

output = np.array(edf_file.createOutput(len(epochs)))

print(inputMatrix.shape)
print(output.shape)

np.random.seed(1)

syn0 = 2*np.random.random((12,8)) - 1
syn1 = 2*np.random.random((8,7)) - 1

for j in range(1000):
    l0 = inputMatrix
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = output - l2
 
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)
 
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
 
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
 
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# start = time.time()
# for e in epochs:
#     e.extractFeatures()
# stop = time.time()
# print("features extracted in: %f sec" % (stop - start))

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
    start = time.time()
    for e in epochs:
        current += 1
        if(verbose==True):
            print("%d / %d" % (current , size))
        e.extractFeatures()
        inputMatrix.append(list(e.features.values()))
    stop = time.time()
    print("features extracted in: %f sec" % (stop - start))
    return np.array(inputMatrix)

def whatStage(output):
    for o in output:
        print(np.argmax(o))

def learn(inputMatrix, output):
    np.random.seed(1)

    syn0 = 2*np.random.random((12,8)) - 1
    syn1 = 2*np.random.random((8,7)) - 1
    
    times = 1
    for j in range(times):
        # print("%d / %d" % (j , times))

        l0 = nonlin(inputMatrix)
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))

        # how much did we miss the target value?
        l2_error = output - l2

        print("output")
        print(output[0])
        print("l2")
        print(l2[0])
        print("l2_error")
        print(l2_error[0])

        # if (j% 200) == 0:
        #     print(l2_error)
    
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * nonlin(l2,deriv=True)

        print("##")
        print(l2_error.shape)
        print("##")
        print("l2_delta")
        print(l2_delta[0])

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
    
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
    
        syn1 += np.dot(l1.T, l2_delta)
        syn0 += np.dot(l0.T, l1_delta)

        print("##")
        print(l1.T.shape)
        print("##")
        print("##")
        print(l2_delta.shape)
        print("##")
        
    return l2


from EdfFile import EdfFile
import numpy as np
import time

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def learn(inputMatrix, output):
    np.random.seed(1)

    syn0 = 2*np.random.random((12,8)) - 1
    syn1 = 2*np.random.random((8,7)) - 1
    
    times = 20000
    for j in range(times):
        print("%d / %d" % (j , times))

        l0 = inputMatrix
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))

        # how much did we miss the target value?
        l2_error = output - l2

        # if (j% 200) == 0:
        #     print(l2_error)
    
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
    
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
    
        syn1 += np.dot(l1.T, l2_delta)
        syn0 += np.dot(l0.T, l1_delta)

    return l2

def compareInOut(outputs, expected):
    difference = expected - outputs
    return np.sum(np.abs(difference))

def accuracy(output, expected):
    hit = 0
    for a, b in zip(output, expected):
        if np.argmax(a) == np.argmax(b):
            hit += 1
    print("%d / %d" % (hit, len(output)))
    # print(np.sum(np.abs(expected[1350] - output[1350])))

def createInput(epochs, verbose=False):
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


from EdfFile import EdfFile
import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def learn(inputMatrix, output, times = 2000):
    np.random.seed(1)

    syn0 = 2*np.random.random((12,10)) - 1
    syn1 = 2*np.random.random((10,8)) - 1
    syn2 = 2*np.random.random((8,7)) - 1

    for j in range(times):
        # print("%d / %d" % (j , times))

        l0 = np.tanh(inputMatrix)
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        l3 = nonlin(np.dot(l2,syn2))

        # how much did we miss the target value?
        l3_error = output - l3
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l3_delta = l3_error * nonlin(l3, deriv=True)

        if (j% (times / 20)) == 0:
            print ("Error:" + str(np.mean(np.abs(l3_error))))

        l2_error = l3_delta.dot(syn2.T)    
        l2_delta = l2_error * nonlin(l2,deriv=True)

        # if (j% 20) == 0:
        #     print("######")
        #     print(l2_error[1])

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
    
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)

        # if (j% 20) == 0:
        #     print("######")
        #     print(l1_error[1])
    
        syn2 += np.dot(l2.T, l3_delta)
        syn1 += np.dot(l1.T, l2_delta)
        syn0 += np.dot(l0.T, l1_delta)

    return l3

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

def countStages(output):
    stages = [0,0,0,0,0,0,0]
    for o in output:
        stages[np.argmax(o)] += 1
    print(stages)


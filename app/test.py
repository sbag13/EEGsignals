# import numpy as np
import statistics
i = 0
stage1 = []
stage2 = []
stage3 = []
stage4 = []
while i < len(epochs):
    if output[i][0] == 1:
        stage1.append(inputMatrix[i])
    if output[i][1] == 1:
        stage2.append(inputMatrix[i])
    if output[i][2] == 1:
        stage3.append(inputMatrix[i])
    if output[i][3] == 1:
        stage4.append(inputMatrix[i])
    i += 1

# st1array = np.array(stage1)
# st1devs = np.array([statistics.stdev(row) for row in st1array.T.tolist()])
# print(st1devs)
# st1mean = np.array([np.mean(row) for row in st1array.T.tolist()])
# print(st1mean)

# print((st1mean - st1devs) / st1meanstage)

# st2array = np.array(stage2)
# st2devs = np.array([stdev(row) for row in st2array.T.tolist()])
# print(st2devs)


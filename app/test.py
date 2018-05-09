import numpy as np
i = 0
stage1 = []
stage2 = []
stage3 = []
while i < len(epochs):
    if output[i][0] == 1:
        stage1.append(inputMatrix[i])
    if output[i][1] == 1:
        stage2.append(inputMatrix[i])
    if output[i][2] == 1:
        stage3.append(inputMatrix[i])
    i += 1

one = np.array([0] * 12)
two = np.array([0] * 12)
three = np.array([0] * 12)

for inn in stage1:
    one = one + inn

for innn in stage2:
    two = two + innn

for innnn in stage3:
    three = three + innnn

one / len(stage1)
two / len(stage2)
three / len(stage3)

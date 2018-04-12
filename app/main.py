from EdfFile import EdfFile
import time

edf_file = EdfFile("../SC4001E0-PSG.edf")
epochs = edf_file.signals_list[1].getEpochs()
# epochs[0].extractFeatures()
start = time.time()
for e in epochs:
    e.extractFeatures()
stop = time.time()
print(stop - start)
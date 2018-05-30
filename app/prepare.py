import numpy as np
import EdfFile
import NN
import Signal
import glob


def prepareFromFile(file_path, annotations):
    edf_file = EdfFile.EdfFile(file_path, annotations)
    epochs = edf_file.signals_list[1].getEpochs()
    outputMatrix = edf_file.createOutput(epochs)

    for e, o in zip(epochs, outputMatrix):
        if max(o) == 0:
            epochs.remove(e)
            outputMatrix.remove(o)

    inputMatrix = edf_file.createInput(epochs, True)

    return inputMatrix, np.array(outputMatrix)

def prepareAllSamples(directory):
    inputMatrix = []
    outputMatrix = []
    psg_files = glob.glob(directory + "/*PSG.edf")
    annotations_files = glob.glob(directory + "/*Hypnogram.edf")
    counter = 1
    for f in psg_files:
        print("(%d / %d) extracting features from file: %s" % (counter, len(psg_files),f))
        annotations = [i for i in annotations_files if i.startswith(f[:-10])]
        in_tmp, out_tmp = prepareFromFile(f, annotations[0])
        inputMatrix.extend(in_tmp)
        outputMatrix.extend(out_tmp)
        counter+=1
    
    return inputMatrix, outputMatrix

# TODO 
# może serializowanie extracted epochs
# zapisywanie stanów pośrednich
# trening!!!
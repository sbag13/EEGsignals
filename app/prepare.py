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

    if (len(inputMatrix) != len(outputMatrix)):
        print("shapes dont match in file %s. input: %d, output: %d" % (file_path, len(inputMatrix), len(outputMatrix)))
        return np.array([]), np.array([])

    return inputMatrix, np.array(outputMatrix)

def prepareAllSamples(directory):
    inputMatrix = []
    outputMatrix = []
    psg_files = glob.glob(directory + "/*PSG.edf")
    annotations_files = glob.glob(directory + "/*Hypnogram.edf")
    counter = 1
    for f in psg_files:         # TODO kilka wątków
        print("(%d / %d) extracting features from file: %s" % (counter, len(psg_files),f))
        annotations = [i for i in annotations_files if i.startswith(f[:-10])]
        in_tmp, out_tmp = prepareFromFile(f, annotations[0])
        inputMatrix.extend(in_tmp)
        outputMatrix.extend(out_tmp)
        counter+=1
    
    return inputMatrix, outputMatrix

def prepareAndSave(fromDirectory, toDirectory):
    inM, outM = prepareAllSamples(fromDirectory)
    np.save(toDirectory + "/inputMatrix", inM)
    np.save(toDirectory + "/outputMatrix", outM)
    print("Successfully saved")

def prepareTraining():
    prepareAndSave("./training_samples", "./training_data")

def preparePredicting():
    prepareAndSave("./predicting_samples", "./predicting_data")

def loadData(directory):
    inM = np.load(directory + "/inputMatrix.npy")
    outM = np.load(directory + "/outputMatrix.npy")
    print("Successfully load")
    return inM, outM

def loadTrainingData():
    return loadData("./training_data")

def loadPredictingData():
    return loadData("./predicting_data")

# TODO 
# może serializowanie extracted epochs
# zapisywanie stanów pośrednich
# trening!!!
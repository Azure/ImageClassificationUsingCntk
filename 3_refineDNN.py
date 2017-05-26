# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


################################################
# MAIN
################################################
# If classifier is set to svm, then no need to run any training iterations
makeDirectory(workingDir)
if classifier == 'svm':
    rf_maxEpochs = 0

# Load data
lutLabel2Id  = readPickle(lutLabel2IdPath)
lutId2Label  = readPickle(lutId2LabelPath)
imgDictTest  = readPickle(imgDictTestPath)
imgDictTrain = readPickle(imgDictTrainPath)
imgDictAug   = readPickle(imgDictAugPath)

# Generate list of active learning images if provided
imgDictAl = {}
for subdir in list(imgDictTrain.keys()):
    imgDictAl[subdir] = getFilesInDirectory(pathJoin(imgAlDir,subdir), ".jpg")
writePickle(imgDictAlPath,  imgDictAl)

# Generate cntk test and train data, i.e. (image, label) pairs and write
# them to disk since in-memory passing is currently not supported by cntk
dataTest  = getImgLabelList(imgDictTest,  imgOrigDir, lutLabel2Id)
dataTrain = getImgLabelList(imgDictTrain, imgOrigDir, lutLabel2Id)
dataAug   = getImgLabelList(imgDictAug,   imgAugDir,  lutLabel2Id)
dataAl    = getImgLabelList(imgDictAl,    imgAlDir,   lutLabel2Id)
dataTrain += dataAug
dataTrain += dataAl

# Optionally add duplicates to balance dataset.
# Note: this should be done using data point weighting (as is done for svm training), rather than using explicit duplicates.
if rf_boBalanceTrainingSet:
    dataTrain = cntkBalanceDataset(dataTrain)

# Print training statistics
print("Statistics training data:")
counts = collections.Counter(getColumn(dataTrain,1))
for label in range(max(lutLabel2Id.values())+1):
    print("   Label {:10}({}) has {:4} training examples.".format(lutId2Label[label], label, counts[label]))

# Train model
# Note: Currently CNTK expects train/test splits to be provided as actual file, rather than in-memory
printDeviceType()
writeTable(cntkTestMapPath,  dataTest)
writeTable(cntkTrainMapPath, dataTrain)
model = train_model(cntkPretrainedModelPath, cntkTrainMapPath, cntkTestMapPath, rf_inputResoluton,
                    rf_maxEpochs, rf_mbSize, rf_maxTrainImages, rf_lrPerMb, rf_momentumPerMb, rf_l2RegWeight,
                    rf_dropoutRate, rf_boFreezeWeights)
model.save(cntkRefinedModelPath)
print("Stored trained model at %s" % cntkRefinedModelPath)

print("DONE. Showing DNN accuracy vs training epoch plot.")
plt.show() # Accuracy vs training epochs plt
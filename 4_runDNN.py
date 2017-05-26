# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


################################################
# MAIN
################################################
# Init
printDeviceType()
makeDirectory(workingDir)
node  = getModelNode(classifier)
model = load_model(cntkRefinedModelPath)
mapPath = pathJoin(workingDir, "rundnn_map.txt")

# Compute dnn output for each image and write to disk
print("Running DNN for test set..")
dnnOutputTest  = runCntkModelAllImages(model, readPickle(imgDictTestPath),  imgOrigDir, mapPath, node, run_mbSize)
print("Running DNN for training set..")
dnnOutputTrain = runCntkModelAllImages(model, readPickle(imgDictTrainPath), imgOrigDir, mapPath, node, run_mbSize)
print("Running DNN for augmentation set..")
dnnOutputAug   = runCntkModelAllImages(model, readPickle(imgDictAugPath),   imgAugDir,  mapPath, node, run_mbSize)
print("Running DNN for active learning set..")
dnnOutputAl    = runCntkModelAllImages(model, readPickle(imgDictAlPath),    imgAlDir,   mapPath, node, run_mbSize)

# Combine all dnn outputs
dnnOutput = dict()
for label in list(dnnOutputTrain.keys()):
    out1 = dnnOutputTrain[label]
    out2 = dnnOutputTest[label]
    out3 = dnnOutputAug[label]
    out4 = dnnOutputAl[label]
    dnnOutput[label] = mergeDictionaries(mergeDictionaries(mergeDictionaries(out1, out2), out3), out4)

# Check if all DNN outputs are of expected size
for label in list(dnnOutput.keys()):
    for feat in list(dnnOutput[label].values()):
        assert(len(feat) == rf_modelOutputDimension)

# Save dnn output to file
print("Writting CNTK outputs to file %s ..." % dnnOutputPath)
writePickle(dnnOutputPath, dnnOutput)
print("DONE.")

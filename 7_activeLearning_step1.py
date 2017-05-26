# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Main
####################################
print("Classifier = " + classifier)
lutLabel2Id = readPickle(lutLabel2IdPath)
mapPath = pathJoin(workingDir, "active_learning_map.txt")

# Get list of images
imgDict = {}
imgDict["unlabeled"] = getFilesInDirectory(imgUnlabeledDir, ".jpg")
assert len(imgDict["unlabeled"])>0, "Need to provide at least 1 image for active learning i in directory: " + imgUnlabeledDir

# Run DNN / SVM classifiers
print("Running DNN...")
node  = getModelNode(classifier)
model = load_model(cntkRefinedModelPath)
dnnOutput = runCntkModelAllImages(model, imgDict, imgRootDir, mapPath, node, run_mbSize)
writePickle(al_dnnOutputPath, dnnOutput)
print("DONE.")
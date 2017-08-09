# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
import json


####################################
# Parameters
####################################
classifier = 'dnn' #must match the option used for model training
imgPath      = "C:/Users/pabuehle/Desktop/ImageClassificationUsingCntk/images/fashionTexture/original/dotted/12.jpg"
resourcesDir = "C:/Users/pabuehle/Desktop/ImageClassificationUsingCntk/proc/fashionTexture/"

# Do not change
run_mbSize = 1
svm_boL2Normalize = True
cntkRefinedModelPath = pathJoin(resourcesDir, "cntk_refined.model")
lutId2LabelPath      = pathJoin(resourcesDir, "lutId2Label.pickle")
svmPath              = pathJoin(resourcesDir, classifier + ".np")  #only used if classifier is set to 'svm'
workingDir           = pathJoin(resourcesDir, "tmp/")


####################################
# Main
####################################
# Init
print("Classifier = " + classifier)
makeDirectory(workingDir)
if not os.path.exists(cntkRefinedModelPath):
    raise Exception("Model file {} does not exist, likely because the {} classifier has not been trained yet.".format(cntkRefinedModelPath, classifier))
model = load_model(cntkRefinedModelPath)
lutId2Label = readPickle(lutId2LabelPath)

# Run DNN
printDeviceType()
node = getModelNode(classifier)
mapPath = pathJoin(workingDir, "rundnn_map.txt")
dnnOutput  = runCntkModelImagePaths(model, [imgPath], mapPath, node, run_mbSize)

# Predicted labels and scores
scoresMatrix = runClassifierOnImagePaths(classifier, dnnOutput, svmPath, svm_boL2Normalize)
scores    = scoresMatrix[0]
predScore = np.max(scores)
predLabel = lutId2Label[np.argmax(scores)]
print("Image predicted to be '{}' with score {}.".format(predLabel, predScore))

# Create json-encoded string of the model output
outDict = {"label": str(predLabel), "score": str(predScore), "allScores": str(scores), "Id2Labels": str(lutId2Label)}
outJsonString = json.dumps(outDict)
print("Json-encoded detections: " + outJsonString[:120] + "...")
print("DONE.")
# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)

####################################
# Parameters
####################################
# No need to change below parameters
boEvalOnTrainingSet = False  # Set to 'False' to evaluate using test set; 'True' to instead eval on training set


####################################
# Main
####################################
print("Classifier = " + classifier)

# Load data
print("Loading data...")
dnnOutput   = readPickle(dnnOutputPath)
lutLabel2Id = readPickle(lutLabel2IdPath)
lutId2Label = readPickle(lutId2LabelPath)
if not boEvalOnTrainingSet:
    imgDict = readPickle(imgDictTestPath)
else:
    print("WARNING: evaluating on training set.")
    imgDict = readPickle(imgDictTrainPath)

# Predicted labels and scores
scoresMatrix, imgFilenames, gtLabels = runClassifier(classifier, dnnOutput, imgDict,  lutLabel2Id, svmPath, svm_boL2Normalize)
predScores = [np.max(scores)    for scores in scoresMatrix]
predLabels = [np.argmax(scores) for scores in scoresMatrix]

# Compute confustion matrix and precision recall curve
confMatrix = confusion_matrix(gtLabels, predLabels)
classes = [lutId2Label[i] for i in range(len(lutId2Label))]
(precisionVec, recallVec, auc) = prComputeCurves(gtLabels, scoresMatrix)
cmPrintAccuracies(confMatrix, classes)

# Plot
plt.figure(figsize=(1,2))
plt.subplot(121)
prPlotCurves(precisionVec, recallVec, auc)
plt.subplot(122)
cmPlot(confMatrix, classes=classes, normalize=True)
plt.draw()
plt.show()

# Visualize results
for counter, (gtLabel, imgFilename, predScore, predLabel) in enumerate(zip(gtLabels, imgFilenames, predScores, predLabels)):
    if predLabel == gtLabel:
        drawColor = (0, 255, 0)
    else:
        drawColor = (0, 0, 255)
    img = imread(pathJoin(imgOrigDir, lutId2Label[gtLabel], imgFilename))
    cv2.putText(img, "{} with score {:2.2f}".format(lutId2Label[predLabel], predScore), (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, drawColor, 2)
    drawCircle(img, (50, 50), 40, drawColor, -1)
    imshow(img, maxDim = 800, waitDuration=500)
print("DONE.")
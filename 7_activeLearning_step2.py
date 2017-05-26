# -*- coding: utf-8 -*-
from helpers import *
#from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
imgOutRootDir = pathJoin(resultsDir, "activeLearning")


####################################
# Main
####################################
# Load data
dnnOutput = readPickle(al_dnnOutputPath)
lutId2Label = readPickle(lutId2LabelPath)
if pathExists(imgOutRootDir):
    for subdir in getDirectoriesInDirectory(imgOutRootDir):
        assert len(getFilesInDirectory(pathJoin(imgOutRootDir,subdir))) == 0, "ERROR: output directory is not empty."

# Predicted labels and scores
# Example svm scoresMatrix row: array([ 0.08620614, -1.1153569 , -0.08649196])
# Example dnn scoresMatrix row: array([ 0.40908358, -0.74543875,  5.61107969])
scoresMatrix, imgFilenames, _ = runClassifier(classifier, dnnOutput, [], [], svmPath, svm_boL2Normalize)
predLabels = [np.argmax(scores) for scores in scoresMatrix]
predLabels = [lutId2Label[i] for i in predLabels]
predScores = [np.max(scores)    for scores in scoresMatrix]
scoresMatrixNormalized = np.array([(row - min(row)) / sum(row - min(row)) for row in scoresMatrix])
predScoresNormalized   = [np.max(scores) for scores in scoresMatrixNormalized]

# Select which images to annotate first
print("Selection criteria = " + al_selectionCriteria)
if al_selectionCriteria == "inorder":
    imgIndices = range(len(predScores))
    imgScores = predScores
if al_selectionCriteria == "random":
    imgIndices = randomizeList(list(range(len(predScores))))
    imgScores = [predScores[i] for i in imgIndices]
elif al_selectionCriteria == "lowestScore":
    imgScores, imgIndices = listSort(np.abs(predScores))
elif al_selectionCriteria == "heighestScore":
    imgScores, imgIndices = listSort(np.abs(predScores), reverseSort=True)
elif al_selectionCriteria == "lowestScoreNormalized":
    imgScores, imgIndices = listSort(np.abs(predScoresNormalized))
elif al_selectionCriteria == "heighestScoreNormalized":
    imgScores, imgIndices = listSort(np.abs(predScoresNormalized), reverseSort=True)
else:
    raise Exception("ERROR: selection criteria unknown: " + al_selectionCriteria)

# Load previously provided annotation (if exists)
if pathExists(al_annotationsPath):
    print("Loading existing annotations from: " + al_annotationsPath)
    annotations = readPickle(al_annotationsPath)
else:
    print("No previously provided annotations found.")
    annotations = dict()

#-----------------------------------------------------------------------------------------------------------------------

# Instructions
print("""
USAGE:
   Press a key at each image, where key can be:
   - Any number in range [1-9] to assign the corresponding label.
   - 'n' to go to the next image without providing any annotation.
   - 'i' to assign the label 'ignore'.
   - 'b' to go back to the previous image without changing any annotations.
   - 'q' to quit this annotation tool.
""")

# Get human annotation for each image
key = ""
loopImgIndex = 0
historyLoopImgIndex = []

while loopImgIndex < len(imgIndices) and key != 'q':
    imgIndex    = imgIndices[loopImgIndex]
    imgScore    = imgScores[loopImgIndex]
    imgFilename = imgFilenames[imgIndex]
    imgPath     = os.path.join(imgUnlabeledDir, imgFilename)

    # check if label was already provided during previous annotation
    if imgFilename in annotations:
        humanLabel = annotations[imgFilename]
        if key != 'b': #only go to next image if user did not just go back to previous image
            loopImgIndex += 1
            print("Skipping image {:4} since already annotated as {:<10}: {}.".format(loopImgIndex, humanLabel, imgFilename))
            continue
    else:
        humanLabel = 'unlabeled'

    #show image
    img, _ = imresizeMaxDim(imread(imgPath), 800, boUpscale = True)
    title = "Score={:2.2f}, Label={}".format(imgScore, humanLabel)
    title += " --- " + ", ".join(["{}={}".format(i+1, lutId2Label[i]) for i in sorted(list(lutId2Label.keys()))])
    cv2.destroyAllWindows()
    cv2.imshow(title, img)

    #parse user input
    historyLoopImgIndex.append(loopImgIndex)
    while (True):
        key = chr(cv2.waitKey())

        # Provided annotation
        if key.isdigit():
            label = lutId2Label[int(key)-1]
            print("Assigning image {} the label: {}".format(imgIndex,label))
            annotations[imgFilename] = label
            #cv2.imshow(imgTitle, np.zeros((100, 100, 3), np.uint8))
            writePickle(al_annotationsPath, annotations)
            loopImgIndex += 1

        # Skip to next image
        elif key == 'n':
            print("Going from image {} to next image.".format(imgIndex))
            loopImgIndex +=1

        # Annotation ignore
        elif key == 'i':
            print("Assigning image {} the label: ignore.".format(imgIndex))
            annotations[imgFilename] = 'ignore'
            writePickle(al_annotationsPath, annotations)
            loopImgIndex += 1

        # Back to previous image
        elif key == 'b':
            if len(historyLoopImgIndex) > 1:
                print("Going back from image {} to previous image.".format(imgIndex))
                loopImgIndex = historyLoopImgIndex[-2]
                historyLoopImgIndex = historyLoopImgIndex[:-2]

        # Quit
        elif key == 'q':
            print("Exiting annotation UI")
            break

        else:
            continue
        break # user input done for this image.
cv2.destroyAllWindows()

#-----------------------------------------------------------------------------------------------------------------------

# Copy all annotated images
print("Copying images")
makeDirectory(imgOutRootDir)
for imgIndex, imgFilename in enumerate(annotations):
    imgLabel = annotations[imgFilename]
    srcPath  = pathJoin(imgUnlabeledDir, imgFilename)
    if imgLabel != 'ignore' and os.path.exists(srcPath):
        dstDir  = pathJoin(imgOutRootDir, imgLabel)
        dstPath = pathJoin(dstDir, imgFilename)
        makeDirectory(dstDir)
        print("Copying image {} of {}: from {} to {}".format(imgIndex, len(annotations), srcPath, dstPath))
        shutil.copy(srcPath, dstPath)
print("DONE.")
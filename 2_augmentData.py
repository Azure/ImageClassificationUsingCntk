# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Main
####################################
random.seed(0)
makeDirectory(imgAugDir)

# Make sure we do not accidentally use already existing augmentation images
for subdir in getDirectoriesInDirectory(imgAugDir):
   assert len(getFilesInDirectory(pathJoin(imgAugDir,subdir))) == 0, "ERROR: Augmentation directory is not empty."

# Compute and save augmentation images
imgDictTrain = readPickle(imgDictTrainPath)
for label in list(imgDictTrain.keys()):
    makeDirectory(pathJoin(imgAugDir, label))
    for imgIndex, imgFilename in enumerate(imgDictTrain[label]):
        augCounter = 0
        printProgressBar(float(imgIndex)/len(imgDictTrain[label]), status="Processing images with label: " + label)

        # rotate image
        for degree in aug_rotationsInDegree:
            if random.random() < aug_probabilityRotation:
                imgPath = pathJoin(imgOrigDir, label, imgFilename)
                imgOrig = imread(imgPath)
                imgRot  = imrotate(imgOrig, degree, Image.BICUBIC, expand = False) #not: this adds a black border. Better would be gray.
                dstFilename = "{}_rot{}.jpg".format(imgFilename[:-4], augCounter, ".jpg")
                dstPath = pathJoin(imgAugDir, label, dstFilename)
                imshow(imgRot, maxDim = 800, waitDuration = 1)
                imwrite(imgRot, dstPath)
                augCounter += 1

# Get list of augmentation images
print("\nStatistics augmentation images:")
imgDictAug = dict()
for subdir in getDirectoriesInDirectory(imgAugDir):
    imgDictAug[subdir] = getFilesInDirectory(pathJoin(imgAugDir, subdir), ".jpg")
    print("   Generated {} images with label {}.".format(len(imgDictAug[subdir]), subdir))
writePickle(imgDictAugPath,  imgDictAug)
print("DONE.")

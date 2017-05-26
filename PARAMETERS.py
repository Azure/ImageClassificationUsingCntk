# -*- coding: utf-8 -*-
import sys, os
from helpers import getDirectoriesInDirectory, pathJoin

#TODO:
# REMOVE unnecessary code
# MOVE HELPERS AND PARAMS TO ACTUALY FUNCTIONS

#######################################
#######################################
datasetName = "fashionTexture"           # Name of the image directory, e.g. /data/myFashion_texture/


###################
# Parameters
###################
classifier = 'svmDnnRefined' #options: 'svm', 'dnn', 'svmDnnRefined'

# Train and test splits (script: 1_prepareData.py)
ratioTrainTest = 0.75                   # Percentage of images used for training of the DNN and the SVM
imagesSplitBy  = 'filename'             # Options: 'filename' or 'subdir'. If 'subdir' is used, then all images in a subdir are assigned fully to train or test

# Augment the training set (script: 2_augmentData.py)
aug_probabilityRotation = 0.2           # Probability with which a rotated image is generated. In range [0,1], where 0 means no added rotation
aug_rotationsInDegree   = [-10, 10]     # Add rotated version of the training images to the training set

# Model refinement parameters (script: 3_refineDNN.py)
rf_pretrainedModelFilename = "ResNet_18.model"  # Pre-trained ImageNet model
rf_inputResoluton = 224                 # DNN image input width and height in pixels. ALso try e.g. 4*224=896 pixels.
rf_dropoutRate    = 0.5                 # Droputout rate
rf_mbSize         = 16                  # Minibatch size (reduce if running out of memory)
rf_maxEpochs      = 5 #45          # Number of training epochs. Set to 0 to skip DNN refinement
rf_maxTrainImages = float('inf')        # Naximum number of training images per epoch. Set to float('inf') to use all images
rf_lrPerMb        = [0.01] * 20 + [0.001] * 20 + [0.0001]  # Learning rate schedule
rf_momentumPerMb  = 0.9                 # Momentum during gradient descent
rf_l2RegWeight    = 0.0005              # L2 regularizer weight during gradient descent
rf_boFreezeWeights      = False         # Set to 'True' to freeze all but the very last layer. Otherwise the full network is refined
rf_boBalanceTrainingSet = False         # Set to 'True' to duplicate images such that all labels have the same number of images

# SVM training params (script: 5_trainSVM.py)
svm_CVals = [10**-4, 10**-3, 10**-2, 0.1, 1, 10, 100] # Slack penality parameter C to try during SVM training

# Active Learning (script: 7_activeLearning_stepX.py)
al_selectionCriteria = "lowestScoreNormalized"

# Running the DNN model (script: 4_runDNN.py and 7_activeLearning_step1.py)
svm_boL2Normalize = True # Normalize 512-floats vector to be of unit length before SVM training
run_mbSize = 64          # Minibatch size when running the model. Higher values will run faster, but might model might not fit into memory



###################
# Fixed parameters
# (do not modify)
###################
print("PARAMETERS: datasetName = " + datasetName)

# Directories
rootDir      = os.path.dirname(os.path.realpath(sys.argv[0])).replace("\\","/") + "/"
imgRootDir   = rootDir + "images/"    + datasetName + "/"
imgOrigDir   = imgRootDir + "original/"
imgAugDir    = imgRootDir + "augmented/"
imgAlDir     = imgRootDir + "activeLearningImages/"
imgUnlabeledDir = imgRootDir + "unlabeled/"
resourcesDir = rootDir + "resources/"
procDir      = rootDir + "proc/"    + datasetName + "/"
resultsDir   = rootDir + "results/" + datasetName + "/"
workingDir   = rootDir + "tmp/"

# Files
dedicatedTestSplitPath  = pathJoin(imgOrigDir, "dedicatedTestImages.tsv")
imgUrlsPath             = pathJoin(resourcesDir, "fashionTextureUrls.tsv")
imgInfosTrainPath       = pathJoin(procDir, "imgInfosTrain.pickle")
imgInfosTestPath        = pathJoin(procDir, "imgInfosTest.pickle")
imgDictTrainPath        = pathJoin(procDir, "imgDictTrain.pickle")
imgDictTestPath         = pathJoin(procDir, "imgDictTest.pickle")
imgDictAugPath          = pathJoin(procDir, "imgDictAug.pickle")
imgDictAlPath           = pathJoin(procDir, "imgDictAl.pickle")
lutLabel2IdPath         = pathJoin(procDir, "lutLabel2Id.pickle")
lutId2LabelPath         = pathJoin(procDir, "lutId2Label.pickle")
if classifier == "svm":
   cntkRefinedModelPath = pathJoin(procDir, "cntk_fixed.model")
else:
   cntkRefinedModelPath = pathJoin(procDir, "cntk_refined.model")
cntkTestMapPath         = pathJoin(workingDir, "test_map.txt")
cntkTrainMapPath        = pathJoin(workingDir, "train_map.txt")
cntkPretrainedModelPath = pathJoin(rootDir, "resources", "cntk", rf_pretrainedModelFilename)
dnnOutputPath           = pathJoin(procDir, "features_" + classifier + ".pickle")
svmPath                 = pathJoin(procDir, classifier + ".np")
al_dnnOutputPath        = pathJoin(procDir, "activeLearning_features.pickle")
al_annotationsPath      = pathJoin(procDir, "activeLearning_annotations.pickle")

# Dimension of the DNN output, for "ResNet_18.model" this is 512 if using a SVM as classifier,
# otherwise the DNN output dimension equals the number of classes
assert(classifier in ['svm', 'dnn', 'svmDnnRefined'])
if classifier.startswith('dnn'):
    rf_modelOutputDimension = len(getDirectoriesInDirectory(imgOrigDir))
elif rf_pretrainedModelFilename.lower() == "resnet_18.model" or rf_pretrainedModelFilename.lower() == "resnet_34.model":
    rf_modelOutputDimension = 512
elif rf_pretrainedModelFilename.lower() == "resnet_50.model":
    rf_modelOutputDimension = 2048
else:
    raise Exception("Model featurization dimension not specified.")
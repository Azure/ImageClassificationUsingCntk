# -*- coding: utf-8 -*-
from helpers import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Main
####################################
random.seed(0)
if not classifier.startswith('svm'):
    print("No need to train SVM since using the DNN directly as classifier.")
    exit()

# Load training datasta
print("Load data...")
lutLabel2Id  = readPickle(lutLabel2IdPath)
imgDictTest  = readPickle(imgDictTestPath)
imgDictTrain = readPickle(imgDictTrainPath)
imgDictAug   = readPickle(imgDictAugPath)
imgDictAl    = readPickle(imgDictAlPath)
dnnOutput  = readPickle(dnnOutputPath)

# Prepare SVM inputs for training and testing
feats_test,  labels_test,  _ = getSvmInput(imgDictTest,  dnnOutput, svm_boL2Normalize, lutLabel2Id)
feats_train, labels_train, _ = getSvmInput(imgDictTrain, dnnOutput, svm_boL2Normalize, lutLabel2Id)
feats_aug,   labels_aug,   _ = getSvmInput(imgDictAug,   dnnOutput, svm_boL2Normalize, lutLabel2Id)
feats_al,    labels_al,    _ = getSvmInput(imgDictAl,    dnnOutput, svm_boL2Normalize, lutLabel2Id)
feats_train  += feats_aug
feats_train  += feats_al
labels_train += labels_aug
labels_train += labels_al
printFeatLabelInfo("Statistics training data:", feats_train, labels_train)
printFeatLabelInfo("Statistics test data:",     feats_test,  labels_test)

# Train SVMs for different values of C, and keep the best result
bestAcc = float('-inf')
for svm_CVal in svm_CVals:
    print("Start SVM training  with C = {}..".format(svm_CVal))
    tstart = datetime.datetime.now()
    #feats_train = sparse.csr_matrix(feats_train) #use this to avoid memory problems
    learner = svm.LinearSVC(C=svm_CVal, class_weight='balanced', verbose=0)
    learner.fit(feats_train, labels_train)
    print("   Training time [labels_train]: {}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))
    print("   Training accuracy    = {:3.2f}%".format(100 * np.mean(sklearnAccuracy(learner, feats_train, labels_train))))
    testAcc = np.mean(sklearnAccuracy(learner, feats_test,  labels_test))
    print("   Test accuracy        = {:3.2f}%".format(100 * np.mean(testAcc)))

    # Store best model. Note that this should use a separate validation set, and not the test set.
    if testAcc > bestAcc:
        print("   ** Updating best model. **")
        bestC = svm_CVal
        bestAcc = testAcc
        bestLearner = learner
print("Best model has test accuracy {:2.2f}%, at C = {}".format(100*bestAcc, bestC))
writePickle(svmPath, bestLearner)
print("Wrote svm to: " + svmPath + "\n")
print("DONE. ")
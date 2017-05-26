# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import map
from builtins import str
from builtins import zip
from builtins import range
from past.builtins import basestring
from past.utils import old_div
from builtins import object

import os, sys, pdb, random, collections, pickle, stat, codecs, itertools, shutil, datetime, importlib, requests
import numpy as np, matplotlib.pyplot as plt
from math import *
from functools import reduce
from sklearn.metrics import *
from os.path import join as pathJoin
from os.path import exists as pathExists

###############################################################################
# Description:
#    This is a collection of general utility / helper functions.
#
# Typical meaning of variable names:
#    lines,strings = list of strings
#    line,string   = single string
#    table         = 2D row/column matrix implemented using a list of lists
#    row,list1D    = single row in a table, i.e. single 1D-list
#    rowItem       = single item in a row
#    list1D        = list of items, not necessarily strings
#    item          = single item of a list1D
#
# Python 2 vs 3:
#    This code was partly automatically converted from python 2 to 3 using 'futurize'.
#    http://python-future.org/compatible_idioms.html
#
# ToDo:
# - port to python3: remove old_div from code where possible
# - see if can remove the list(range()), etc statements introduced by futurize
###############################################################################


#################################################
# File access
#################################################
def readFile(inputFile):
    # Comment from Python 2: reading as binary, to avoid problems with end-of-text
    #    characters. Note that readlines() does not remove the line ending characters
    with open(inputFile,'rb') as f:
        lines = f.readlines()
        #lines = [unicode(l.decode('latin-1')) for l in lines]  convert to uni-code
    return [removeLineEndCharacters(s.decode('utf8')) for s in lines];

def readBinaryFile(inputFile):
    with open(inputFile,'rb') as f:
        bytes = f.read()
    return bytes

def readPickle(inputFile):
    with open(inputFile, 'rb') as filePointer:
         data = pickle.load(filePointer)
    return data

def readTable(inputFile, delimiter='\t', columnsToKeep=None):
    # Note: if getting memory errors then use 'readTableFileAccessor' instead
    lines = readFile(inputFile);
    if columnsToKeep != None:
        header = lines[0].split(delimiter)
        columnsToKeepIndices = listFindItems(header, columnsToKeep)
    else:
        columnsToKeepIndices = None;
    return splitStrings(lines, delimiter, columnsToKeepIndices)

class readFileAccessorBase(object):
    def __init__(self, filePath, delimiter):
        self.fileHandle = open(filePath,'rb')
        self.delimiter = delimiter
        self.lineIndex = -1
    def __iter__(self):
        return self
    def __exit__(self, dummy1, dummy2, dummy3):
        self.fileHandle.close()
    def __enter__(self):
        pass
    def __next__(self):
        self.lineIndex += 1
        line = self.fileHandle.readline()
        line = removeLineEndCharacters(line)
        if self.delimiter != None:
            return splitString(line, delimiter='\t', columnsToKeepIndices=None)
        else:
            return line

class readTableFileAccessor(readFileAccessorBase):
    # Iterator-like file accessor. Usage example: "for line in readTableFileAccessor("input.txt"):"
    def __init__(self, filePath, delimiter = '\t'):
        readFileAccessorBase.__init__(self, filePath, delimiter)

class readFileAccessor(readFileAccessorBase):
    def __init__(self, filePath):
        readFileAccessorBase.__init__(self, filePath, None)

def writeFile(outputFile, lines, header=None, encoding=None):
    if encoding == None:
        with open(outputFile,'w') as f:
            if header != None:
                f.write("%s\n" % header)
            for line in lines:
                f.write("%s\n" % line)
    else:
        with codecs.open(outputFile, 'w', encoding) as f:  # e.g. encoding=utf-8
            if header != None:
                f.write("%s\n" % header)
            for line in lines:
                f.write("%s\n" % line)

def writeTable(outputFile, table, header=None):
    lines = tableToList1D(table)
    writeFile(outputFile, lines, header)

def writeBinaryFile(outputFile, data):
    with open(outputFile,'wb') as f:
        bytes = f.write(data)
    return bytes

def writePickle(outputFile, data):
    p = pickle.Pickler(open(outputFile,"wb"))
    p.fast = True
    p.dump(data)

def getFilesInDirectory(directory, postfix = ""):
    if not os.path.exists(directory):
        return []
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(directory+"/"+s)]
    if not postfix or postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def getFilesInSubdirectories(directory, postfix = ""):
    paths = []
    for subdir in getDirectoriesInDirectory(directory):
        for filename in getFilesInDirectory(os.path.join(directory, imgSubdir), postfix):
            paths.append(os.path.join(directory, subdir, filename))
    return paths

def getDirectoriesInDirectory(directory):
    return [s for s in os.listdir(directory) if os.path.isdir(directory+"/"+s)]

def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def makeOrClearDirectory(directory):
    # Note: removes just the files in the directory, not recursive
    makeDirectory(directory)
    files = os.listdir(directory)
    for file in files:
        filePath = directory +"/"+ file
        os.chmod(filePath, stat.S_IWRITE )
        if not os.path.isdir(filePath):
            os.remove(filePath)

def removeWriteProtectionInDirectory(directory):
    files = os.listdir(directory)
    for file in files:
        filePath = directory +"/"+ file
        if not os.path.isdir(filePath):
            os.chmod(filePath, stat.S_IWRITE )

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def deleteAllFilesInDirectory(directory, fileEndswithString, boPromptUser = False):
    if boPromptUser:
        userInput = eval(input('--> INPUT: Press "y" to delete files in directory ' + directory + ": "))
        if not (userInput.lower() == 'y' or userInput.lower() == 'yes'):
            print("User input is %s: exiting now." % userInput)
            exit()
    for filename in getFilesInDirectory(directory):
        if fileEndswithString == None or filename.lower().endswith(fileEndswithString):
            deleteFile(directory + "/" + filename)



#################################################
# 1D list
#################################################
def isList(var):
    return isinstance(var, list)

def toIntegers(list1D):
    return [int(float(x)) for x in list1D]

def toRounded(list1D):
    return [round(x) for x in list1D]

def toFloats(list1D):
    return [float(x) for x in list1D]

def toStrings(list1D):
    return [str(x) for x in list1D]

def max2(list1D):
    maxVal = max(list1D)
    indices = [i for i in range(len(list1D)) if list1D[i] == maxVal]
    return maxVal,indices

def pbMax(list1D): # depricated
    return max2(list1D)

def find(list1D, func):
    return [index for (index,item) in enumerate(list1D) if func(item)]

def findNearest(list1D, value):
    index = (np.abs(list1D-value)).argmin()
    return list1D[index], index

def listFindItem(list1D, itemToFind):
    return listFindItems(list1D, [itemToFind])

def listFindItems(list1D, itemsToFind):
    indices = [];
    list1DSet = set(list1D)
    for item in itemsToFind:
        if item in list1DSet:
            index = list1D.index(item) #returns first of possibly multiple hits
            indices.append(index)
    return indices

def listFindSublist(list1D, itemToFindList1D):
    # Example: list1D = ['this', 'is', 'a', 'test']; itemToFindList = ['is','a']
    matchIndices = []
    nrItemsToFind = len(itemToFindList1D)
    for startIndex in range(len(list1D)-nrItemsToFind+1):
        endIndex = startIndex + nrItemsToFind -1
        if list1D[startIndex:endIndex+1] == itemToFindList1D:
            matchIndices.append(startIndex)
    return matchIndices

def listFindSubstringMatches(strings, stringsToFind, ignoreCase):
    indices = []
    for (index,string) in enumerate(strings):
        for stringToFind in stringsToFind:
            if ignoreCase:
                string = string.upper()
                stringToFind = stringToFind.upper()
            if string.find(stringToFind) >= 0:
                indices.append(index)
                break
    return indices

def listSort(list1D, reverseSort=False, comparisonFct=lambda x: x):
    indices = list(range(len(list1D)))
    tmp = sorted(zip(list1D,indices), key=comparisonFct, reverse=reverseSort)
    list1DSorted, sortOrder = list(map(list, list(zip(*tmp))))
    return (list1DSorted, sortOrder) 

def listExtract(list1D, indicesToKeep):
    indicesToKeepSet = set(indicesToKeep)
    return [item for index,item in enumerate(list1D) if index in indicesToKeepSet]

def listRemove(list1D, indicesToRemove):
    indicesToRemoveSet = set(indicesToRemove)
    return [item for index,item in enumerate(list1D) if index not in indicesToRemoveSet]

def listReverse(list1D):
    return list1D[::-1]

def listUniquify(list1D):
    uniqueList = []
    uniqueSet = set()
    uniqueIndices = []
    for index,item in enumerate(list1D):
        if item not in uniqueSet:
            uniqueList.append(item)
            uniqueSet.add(item)
            uniqueIndices.append(index)
    return (uniqueList, uniqueIndices)

def listRemoveEmptyStrings(strings):
    indices = find(strings, lambda x: x != "")
    return [strings[i] for i in indices]

def listRemoveEmptyStringsFromEnd(strings):
    while len(strings)>0 and strings[-1] == "":
        strings = strings[:-1]
    return strings

def listIntersection(listA, listB):
    listBSet = set(listB)
    return [item for item in listA if item in listBSet]

def listsIdentical(listA, listB, allowPermutation = False):
    if len(listA) != len(listB):
        return False
    if allowPermutation:
        zipObj = zip(sorted(listA),sorted(listB))
    else:
        zipObj = zip(listA, listB)
    for (elemA, elemB) in zipObj:
        if elemA!=elemB:
            return False
    return True



#################################################
# 2D list (e.g. tables)
#################################################
def getColumn(table, columnIndex):
    return [row[columnIndex] for row in table]

def getRows(table, rowIndices):    
    return [table[rowIndex] for rowIndex in rowIndices]

def getColumns(table, columnIndices):
    return [[row[i] for i in columnIndices] for row in table]

def splitColumn(table, columnIndex, delimiter):
    # Creates a longer table by splitting items of a given row
    newTable = [];
    for row in table:
        items = row[columnIndex].split(delimiter)
        for item in items:
            row = list(row) #make copy
            row[columnIndex] = item
            newTable.append(row)
    return newTable

def sortTable(table, sortColumnIndex, reverseSort=False, comparisonFct=lambda x: float(x[0])):
    if len(table) == 0:
        return []
    list1D = getColumn(table, sortColumnIndex)
    _, sortOrder = listSort(list1D, reverseSort, comparisonFct)
    return [table[i] for i in sortOrder]

def flattenTable(table):
    return [x for x in reduce(itertools.chain, table)]

def tableToList1D(table, delimiter='\t'):
    return [delimiter.join([str(s) for s in row]) for row in table]



#################################################
# String and chars
#################################################
def isAlphabetCharacter(c):
    ordinalValue = ord(c)
    if (ordinalValue>=65 and ordinalValue<=90) or (ordinalValue>=97 and ordinalValue<=122):
        return True
    return False

def isNumberCharacter(c):
    ordinalValue = ord(c)
    if (ordinalValue>=48 and ordinalValue<=57):
        return True
    return False

def isString(var):
    return type(var) == type("")

def numToString(num, length, paddingChar = '0'):
    if len(str(num)) >= length:
        return str(num)[:length]
    else:
        return str(num).ljust(length, paddingChar)

# def numToString(number, nrDecimalDigits):
#     numString = str(round(number,nrDecimalDigits))
#     currNrDecimalDigits = len(numString) -1 -len(str(int(number)))
#     numString += "0" * (nrDecimalDigits-currNrDecimalDigits)
#     return numString

def stringsEqual(string1, string2, ignoreCase=False):
    if ignoreCase:
        string1 = string1.upper()
        string2 = string2.upper()
    return string1 == string2

def findFirstSubstring(string, stringToFind, ignoreCase=False):
    if ignoreCase:
        string = string.upper();
        stringToFind = stringToFind.upper();
    return string.find(stringToFind)

def findMultipleSubstrings(string, stringToFind, ignoreCase=False):
    if ignoreCase:
        string = string.upper();
        stringToFind = stringToFind.upper();
    matchPositions = [];
    pos = string.find(stringToFind)
    while pos >= 0:
        matchPositions.append(pos)
        pos = string.find(stringToFind, pos + 1)
    return matchPositions

def findMultipleSubstringsInMultipleStrings(string, stringsToFind, ignoreCase=False):
    matches = []
    for (stringToFindIndex,stringToFind) in enumerate(stringsToFind):
        matchStartPositions = findMultipleSubstrings(string, stringToFind, ignoreCase)
        for matchStartPos in matchStartPositions:
            matchEndPos = matchStartPos + len(stringToFind)
            matches.append([matchStartPos,matchEndPos,stringToFindIndex])
    return matches

def regexMatch(string, regularExpression, matchGroupIndices):
    regexMatches = re.match(regularExpression, string)
    if regexMatches != None:
        matchedStrings = [regexMatches.group(i) for i in matchGroupIndices]
    else:
        matchedStrings = [None] * len(matchGroupIndices)
    if len(matchGroupIndices) == 1:
        matchedStrings = matchedStrings[0]
    return matchedStrings

def insertInString(string, pos, stringToInsert):
    return insertInString(string, pos, pos, stringToInsert)

def insertInString(string, textToKeepUntilPos, textToKeepFromPos, stringToInsert):
    return string[:textToKeepUntilPos] + stringToInsert + string[textToKeepFromPos:]

def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if string == None:
        return None
    items = string.split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumn(items, columnsToKeepIndices)
    return items;

def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table;

def spliceString(string, textToKeepStartPos, textToKeepEndPos):
    stringNew = "";
    for (startPos, endPos) in zip(textToKeepStartPos,textToKeepEndPos):
        stringNew = stringNew + string[startPos:endPos+1]
    return stringNew

def removeMultipleSpaces(string):
    return re.sub('[ ]+' , ' ', string)

def removeAllSpaces(string):
    return string.replace(" ", "")

def removeLineEndCharacters(line):
    if line.endswith('\r\n'):
        return line[:-2]
    elif line.endswith('\n'):
        return line[:-1]
    else:
        return line

def replaceNthWord(string, wordIndex, wordToReplaceWith):
    words = string.split()
    words[wordIndex] = wordToReplaceWith
    return " ".join(words)

def removeWords(string, wordsToRemove, ignoreCase=False):
    newWords = [word for word in string.split() if not listExists(word, wordsToRemove, ignoreCase)]
    return " ".join(newWords)

def removeNthWord(string, wordIndex):
    words = string.split()
    if wordIndex == 0:
        wordsNew = words[1:]
    elif wordIndex == len(words)-1:
        wordsNew = words[:-1]
    else:
        wordsNew = words[:wordIndex] + words[wordIndex+1:]
    return " ".join(wordsNew)

def containsOnlyRegularAsciiCharacters(string):
    return all(ord(c) < 128 for c in string)

def removeControlCharacters(string):
    # remove all control characters except for TAB
    # see: http://www.asciitable.com/
    chars = [c for c in string if not ((ord(c)>=0 and ord(c)<=8) or (ord(c)>=10 and ord(c)<=31))]
    return "".join(chars)



#################################################
# Randomize
#################################################
def getRandomNumber(low, high):
    return random.randint(low,high)

def getRandomNumbers(low, high):
    randomNumbers = list(range(low,high+1))
    random.shuffle(randomNumbers)
    return randomNumbers

def getRandomListElement(listND, containsHeader=False):
    if containsHeader:
        index = getRandomNumber(1, len(listND)-1)
    else:
        index = getRandomNumber(0, len(listND)-1)
    return listND[index]

def randomizeList(listND, containsHeader=False):
    if containsHeader:
        header = listND[0]
        listND = listND[1:]
    random.shuffle(listND)
    if containsHeader:
        listND.insert(0, header)
    return listND

def subsampleList(listND, maxNrSamples):
    indices = list(range(len(listND)))
    random.shuffle(indices)
    nrItemsToSample = min(len(indices), maxNrSamples)
    return [listND[indices[i]] for i in range(nrItemsToSample)]

def randomSplit(list1D, ratio):
    indices = list(range(len(list1D)))
    random.shuffle(indices)
    nrItems = int(round(ratio * len(list1D)))
    listA = [list1D[i] for i in indices[:nrItems]]
    listB = [list1D[i] for i in indices[nrItems:]]
    return (listA,listB)



#################################################
# Dictionaries
#################################################
def getDictionary(keys, values, boConvertValueToInt = True):
    dictionary = {}
    for key, value in zip(keys, values):
        if boConvertValueToInt:
            value = int(value)
        dictionary[key] = value
    return dictionary

def sortDictionary(dictionary, sortIndex=0, reverseSort=False):
    return sorted(list(dictionary.items()), key=lambda x: x[sortIndex], reverse=reverseSort)

def invertDictionary(dictionary):
    return {v: k for k, v in list(dictionary.items())}

def dictionaryToTable(dictionary):
    return (list(dictionary.items()))

def mergeDictionaries(dict1, dict2):
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp

def increaseDictionary(dictionary, key, addValue=1, initialValue=0):
    if key in list(dictionary.keys()):
        dictionary[key] += addValue
    else:
        dictionary[key] = initialValue + addValue



#################################################
# Collections.Counter()
#################################################
def countItems(list1D):
    counts = collections.Counter()
    for item in list1D:
        counts[item] += 1
    return counts

def countWords(sentences, ignoreCase=True):
    counts = collections.Counter()
    for sentence in sentences:
        for word in sentence.split():
            if ignoreCase:
                word = word.lower()
            counts[word] += 1
    return counts

def convertCounterToList(counter, threshold=None):
    sortedKeyValuePairs = counter.most_common()
    if threshold != None:
        sortedKeyValuePairs = [[key,value] for key,value in sortedKeyValuePairs if value >= threshold]
    return sortedKeyValuePairs



#################################################
# Processes
# (start process using: p = subprocess.Popen(cmdStr))
#################################################
def isProcessRunning(processID):
    status = processID.poll()
    return status == None

def countNumberOfProcessesRunning(processIDs):
    return sum([isProcessRunning(p) for p in processIDs])
    


#################################################
# Arguments
#################################################
def printParsedArguments(options):
    print("Arguments parsed in using the command line:")
    for varName in [v for v in dir(options) if not callable(getattr(options,v)) and v[0] != '_']:
        exec(('print "   %s = "') % varName)
        exec(('print options.%s') % varName)

def optionParserSplitListOfValues(option, value, parser):
    setattr(parser.values, option.dest, value.split(','))



#################################################
# Url
################################################# 
def removeURLPrefix(url, urlPrefixes = ["https://www.", "http://www.", "https://", "http://", "www."]):
    for urlPrefix in urlPrefixes:
        if url.startswith(urlPrefix):
            url = url[len(urlPrefix):]
            break
    return url

def urlsShareSameRoot(url, urlRoot):
    url = removeURLPrefix(url)
    urlRoot = removeURLPrefix(urlRoot)
    return url.startswith(urlRoot)

def downloadFromUrl(url, boVerbose = True):
    data = []
    try:
        r = requests.get(url)
        data = r.content
    except:
        if boVerbose:
            print('Error downloading url {0}'.format(url))
    #if boVerbose and data == []: # and r.status_code != 200:
    #    print('Error {} downloading url {}'.format(r.status_code, url))
    return data



#################################################
# Confusion matrix and p/r curves
#################################################
def cmGetAccuracies(confMatrix):
    return [float(confMatrix[i, i]) / sum(confMatrix[:, i]) for i in range(confMatrix.shape[1])]

def cmPrintAccuracies(confMatrix, classes):
    columnWidth = max([len(s) for s in classes])
    accs = cmGetAccuracies(confMatrix)
    for cls, acc in zip(classes, accs):
        print(("Class {:<" + str(columnWidth) + "} accuracy: {:2.2f}%.").format(cls, 100 * acc))
    print("OVERALL accuracy: {:2.2f}%.".format(100.0 * sum(np.diag(confMatrix)) / sum(sum(confMatrix))))
    print("OVERALL class-averaged accuracy: {:2.2f}%.".format(100 * np.mean(accs)))

def cmPlot(confMatrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        confMatrix = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
        confMatrix = np.round(confMatrix * 100,1)

    #Actual plotting of the values
    thresh = confMatrix.max() / 2.
    for i, j in itertools.product(range(confMatrix.shape[0]), range(confMatrix.shape[1])):
        plt.text(j, i, confMatrix[i, j], horizontalalignment="center",
                 color="white" if confMatrix[i, j] > thresh else "black")

    avgAcc = np.mean([float(confMatrix[i, i]) / sum(confMatrix[:, i]) for i in range(confMatrix.shape[1])])
    plt.imshow(confMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title + " (avgAcc={:2.2f}%)".format(100*avgAcc))
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def prComputeCurves(gtLabels, scoresMatrix):
    predScores   = [np.max(scores) for scores in scoresMatrix]
    predLabels   = np.array([np.argmax(scores) for scores in scoresMatrix])
    predsCorrect = np.array(np.array(predLabels) == np.array(gtLabels), int)
    #sampleWeights = getSampleWeights(gtLabels) # balanca pos and neg examples, currently only works for binary
    auc = average_precision_score(predsCorrect, predScores, average="macro") #, sample_weight = sampleWeights)
    precisionVec, recallVec, _ = precision_recall_curve(predsCorrect, predScores) #, sample_weight = sampleWeights)
    return (precisionVec, recallVec, auc)

def prPlotCurves(precisionVec, recallVec, auc):
    plt.plot(recallVec, precisionVec, color='gold', lw=2, label='Overall system (area = {0:0.2f})'.format(auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend(loc="lower right")
    #plt.tight_layout()



#################################################
# Math
#################################################
def intRound(item):
    return int(round(float(item)))

def softmax(vec):
    expVec = np.exp(vec)
    if max(expVec) != np.inf:
        outVec = expVec / np.sum(expVec)
    else:
        # Note: this is a hack to make softmax stable
        outVec = np.zeros(len(expVec))
        outVec[expVec == np.inf] = vec[expVec == np.inf]
        outVec = outVec / np.sum(outVec)
    return outVec

def softmax2D(w):
    # Note: replace with np.exp(w – max(w)) to make numerically stable?
    e = np.exp(w)
    dist = old_div(e, np.sum(e, axis=1)[:, np.newaxis])
    return dist

def chiSquared(vec1, vec2, eps = 10**-8):
    return sum(np.square(vec1 - vec2) / (vec1 + vec2 + eps))

def computeVectorDistance(vec1, vec2, method, boL2Normalize, weights = [], bias = []):
    assert (len(vec1) == len(vec2))
    if boL2Normalize:
        vec1 /= np.linalg.norm(vec1, 2)
        vec2 /= np.linalg.norm(vec2, 2)

    # Distance computation
    vecDiff = vec1 - vec2
    method = method.lower()
    if method == 'random':
        dist = random.random()
    elif method == 'l1':
        dist = sum(abs(vecDiff))
    elif method == 'l2':
        dist = np.linalg.norm(vecDiff, 2)
    elif method == 'normalizedl2':
        a = vec1 / np.linalg.norm(vec1, 2)
        b = vec2 / np.linalg.norm(vec2, 2)
        dist = np.linalg.norm(a - b, 2)
    elif method == "cosine":
        dist = scipy.spatial.distance.cosine(vec1, vec2)
    elif method == "correlation":
        dist = scipy.spatial.distance.correlation(vec1, vec2)
    elif method == "chisquared":
        dist = chiSquared(vec1, vec2)
    elif method == "normalizedchisquared":
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        dist = chiSquared(a, b)
    elif method == "hamming":
        dist = scipy.spatial.distance.hamming(vec1 > 0, vec2 > 0)
    elif method == "mahalanobis":
        # Assumes covariance matric is provided, eg. using: sampleCovMat = np.cov(np.transpose(np.array(feats)))
        dist = scipy.spatial.distance.mahalanobis(vec1, vec2, sampleCovMat)
    elif method == 'weightedl1':
        feat = np.float32(abs(vecDiff))
        dist = np.dot(weights, feat) + bias
        dist = -float(dist)
    elif method == 'weightedl2':
        feat = (vecDiff) ** 2
        dist = np.dot(weights, feat) + bias
        dist = -float(dist)
    # elif method == 'learnerprob':
    #     feat = (vecDiff) ** 2
    #     dist = learner.predict_proba([feat])[0][1]
    #     dist = float(dist)
    # elif method == 'learnerscore':
    #     feat = (vecDiff) ** 2
    #     dist = learner.base_estimator.decision_function([feat])[0]
    #     dist = -float(dist)
    else:
        raise Exception("Distance method unknown: " + method)
    assert (not np.isnan(dist))
    return dist

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    # Compute VOC AP given precision and recall. See also Faster-RCNN github repo
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrecalls    = np.concatenate(([0.], recalls,    [1.]))
        mprecisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        # and sum (\Delta recall) * prec
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap



#################################################
# other
#################################################
def isTuple(var):
    return isinstance(var, tuple)

def printProgressMsg(msgFormatString, currentValue, maxValue, modValue):
    if currentValue % modValue == 1:
        text = "\r" + msgFormatString.format(currentValue, maxValue)
        sys.stdout.write(text)
        sys.stdout.flush()

def printProgressBar(progress, status = ""):
    barLength = 30
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#"*block + "-"*(barLength-block), round(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()

def showVars(context, maxNrLines = 10**6):
    # Note: this does not always show the correct variable size
    varSizes = []
    varsDictItems = list(context.items())
    for varsDictItem in varsDictItems:
        obj = varsDictItem[1]
        if type(obj) is list:
            size = 0
            for listItem in obj:
                try:
                    size += listItem.nbytes
                except:
                    size += sys.getsizeof(listItem)
        else:
            size = sys.getsizeof(obj)
        varSizes.append(size)
    _, sortOrder = listSort(varSizes, reverseSort = True)
    print("{0:10} | {1:30} | {2:100}".format("SIZE", "TYPE", "NAME"))
    print("="*100)
    for index in sortOrder[:maxNrLines]:
        print("{0:10} | {1:30} | {2:100}".format(varSizes[index], type(varsDictItems[index][1]), varsDictItems[index][0]))


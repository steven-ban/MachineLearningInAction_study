#!/usr/bin/python
# coding = utf-8
# decison tree
# use shannon Ent to calculate information gain

from math import log
import operator

def calcShannonEnt(dataSet):
	# create diction for every possible entry
	numEntries = len(dataSet)	# number of rows
	labelCounts = {}
	for featVec in dataSet:	# travel all the samples, calculae the number of "yes" and "no"
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	
	# calc Shannon Ent
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
		
	return shannonEnt
	
# define data set
def createDataSet():
	dataSet = [[1, 1, 'yes'], \
		[1, 1, 'yes'], \
		[1, 0, 'no'], \
		[0, 1, 'no'], \
		[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


# split the data set 
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis + 1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet
	
	
# choose best data set split
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	# initialize var
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)	# create unique split label list
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
	

def majortyCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritem(),\
			key = operator.itemgetter(1), \
			reverse = True)
	return sortedClassCount[0][0]
	
# create decision tree
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majortyCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree
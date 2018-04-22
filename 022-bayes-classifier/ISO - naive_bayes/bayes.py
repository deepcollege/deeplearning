import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = np.array(pd.read_csv('pima-indians-diabetes.data.csv.txt'))

train, test = train_test_split(data, test_size=0.2)


# seperates last element in row of matrix into dict
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		row = dataset[i]
		if (row[-1] not in separated):
			separated[row[-1]] = []
		separated[row[-1]].append(row)
	return separated


def summarize(dataset):
	summaries = [(np.mean(attribute), np.std(attribute, ddof=1)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries


def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries


def calculateProbability(x, mean, stdev):
	exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
	return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities




def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet)))
if __name__ == '__main__':
	summaries = summarizeByClass(train)
	# test model
	predictions = getPredictions(summaries, test)
	accuracy = getAccuracy(test, predictions)
	print('Accuracy:', accuracy)
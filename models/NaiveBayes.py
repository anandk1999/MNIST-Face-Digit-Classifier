import numpy as np
import time
import sys
sys.path.append(r'/Users/ASUS1/Desktop/Spring 2023 Semester/Introduction to Artificial Intelligence/Assignment 1/CS440/Final Project')
from helpers import featureExctractor, constants, getData, util, statisticsWriter

class NaiveBayesClass:
    def __init__(self, n_iters=3):
        self.rawTrainingData = None
        self.trainingLabels = None
        self.rawValidationData = None
        self.validationLabels = None
        self.rawTestData = None
        self.testLabels = None
        self.legalLabels = None

        self.trainingData = None
        self.validationData = None
        self.testData = None
        
        self.n_iters = n_iters

        self.statistics = {}
    
    
    def getFeatures(self, dataType = 'd'):
        if dataType == 'd':
            self.trainingData = list(map(featureExctractor.Digit, self.rawTrainingData))
            self.testData = list(map(featureExctractor.Digit, self.rawTestData))
        elif dataType == 'f':
            self.trainingData = list(map(featureExctractor.Face, self.rawTrainingData))
            self.testData = list(map(featureExctractor.Face, self.rawTestData))
        
        return True
    
    def prior_distribution(self, label, type):
        if type == 'train':
            return sum(1 for item in self.trainingLabels if item == label)/len(self.trainingLabels)
        else:
            return sum(1 for item in self.testLabels if item == label)/len(self.testLabels)
    
    def conditional_probability(self, pixel, val, label, type):
        if type == 'train':
            indices = [i for i, x in enumerate(self.trainingLabels) if x==label]

            successes = 0
            total = 1
            for i in indices:
                if self.trainingData[i][pixel]==val:
                    successes+=1
            if val==0:
                total+=successes
                for i in indices:
                    if self.trainingData[i][pixel]==1:
                        total+=1
            else:
                total+=successes
                for i in indices:
                    if self.trainingData[i][pixel]==0:
                        total+=1
            prob = successes/total
            return prob
        else:
            indices = [i for i, x in enumerate(self.testLabels) if x==label]

            successes = 0
            total = 1
            for i in indices:
                if self.testData[i][pixel]==val:
                    successes+=1
            if val==0:
                total+=successes
                for i in indices:
                    if self.testData[i][pixel]==1:
                        total+=1
            else:
                total+=successes
                for i in indices:
                    if self.testData[i][pixel]==0:
                        total+=1
            prob = successes/total
            return prob
    def log_probability(self, datum, label, type):
        final_prob = np.log(self.prior_distribution(label, type))

        cond_prob = 0
        for key,val in datum.items():
            cond_prob+=np.log(self.conditional_probability(key,val,label,type))

        final_prob += cond_prob

        return final_prob

    def classify(self, data, type):
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.log_probability(datum,l, type)
            guesses.append(vectors.argMax())
        return guesses
    
    def train(self, iter=3):
        mean_acc = 0
        for iteration in range(iter):
            accuracy = []
            for i in range(len(self.trainingData)):

                actual = self.trainingLabels[i]
                bestGuess = self.classify([self.trainingData[i]], 'train')[0]

                if bestGuess==actual:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
            mean_acc+= sum(accuracy)/len(accuracy)
        return mean_acc/3
            
    
    def test(self):
        self.predictions = self.classify(self.testData, 'test')
        return self.predictions

    def run(self, iters = 5, debug=False):

        self.testIters = iters

        if debug:
            TRAINDATA_SIZE = [0.1]
        else:
            TRAINDATA_SIZE = np.arange(0.1, 1.1, 0.1)
        
        dataTypes = ['d', 'f']# Digits and Faces.

        for dataType in dataTypes:

            self.statistics[dataType] = {}

            if dataType == 'f':
                dataSize = constants.FACE_TRAINING_DATA_SIZE
            else:
                dataSize = constants.DIGITS_TRAINING_DATA_SIZE

            for size in TRAINDATA_SIZE:

                acc = []
                avgTime = []

                for index in range(self.testIters):
                    trainingSize = int(size * dataSize)

                    # Fetch Data.
                    [self.rawTrainingData, 
                    self.trainingLabels,  
                    self.rawTestData, 
                    self.testLabels,
                    self.legalLabels] = getData.fetch(dataType, trainingSize)

                    # Convert raw data into features we want.
                    self.getFeatures(dataType=dataType)

                    print(f'Naive Bayes Training with {dataType} data and size {trainingSize}[{int(size*100)}%] and iteration {index}.....')
                    testStart = time.time()
                    test_accuracy = self.train()
                    testTime = time.time() -  testStart
                    avgTime.append(testTime)

                    print(f'Naive Bayes Testing Accuracy with {dataType} data and size {trainingSize}: {test_accuracy*100}%')

                    print(f'Naive Bayes Testing with {dataType} data and size {trainingSize}[{int(size*100)}%] and iteration {index}.....')
                    preds = self.test()

                    acc.append([preds[i] == self.testLabels[i] for i in range(len(self.testLabels))].count(True) / len(self.testLabels))

                    print(f'Naive Bayes Prediction Accuracy with {dataType} data and size {trainingSize}[{int(size*100)}%]: {acc[index] * 100}[iteration {index}]')
            
                # Once we have finished iterations on 10%, 20%, 30%....
                acc = np.array(acc)
                avgTime = np.array(avgTime)
                
                self.statistics[dataType][int(size*100)] = {}
                self.statistics[dataType][int(size*100)]['mean'] = np.mean(acc)
                self.statistics[dataType][int(size*100)]['std'] = np.std(acc)
                self.statistics[dataType][int(size*100)]['avgTime'] = np.mean(avgTime)
            
            print()
            
        return self.statistics
    
    def write(self):
        statisticsWriter.write(self.statistics, self.testIters)

# Testing Process.
if __name__ == '__main__':
    classifierOne = NaiveBayesClass()
    classifierOne.run(debug=True)
    classifierOne.write()

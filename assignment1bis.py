#-------------------------------------------------------------------------------
# Name:        CSC492 - Coding Assignment #1
# Purpose:
#
# Author:      Marion
#
# Created:     13/10/2017
# Copyright:   (c) Marion 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import csv
import numpy as np
from numpy.random import randn

#seed the random numbers to help debbugging

np.random.seed(1)

#define hyperparameters

LEARNING_RATE = 0.01
NB_FEATURES = 26
NB_TRAININGEX = 13000
NB_CLASSES = 13
NB_HIDDEN_NEURONS = 16
NB_TEST = 10400

#getting the test data from the csv file
testDatatemp = np.loadtxt(open("test.x.csv","rb"), dtype =np.float16,delimiter = ',',skiprows=1, usecols=range(1,27))

#getting the input data from the csv file
input = np.loadtxt(open("train.x.csv","rb"), dtype =np.float16,delimiter = ',',skiprows=1, usecols=range(1,27))

#adding a column of ones for the bias term in the input training array
inputBias = np.ones((NB_TRAININGEX, 1))
inputFinal = np.hstack((inputBias,input))

print testDatatemp.shape
#adding a column of ones for the bias term in the input testing array
testBias = np.ones((NB_TEST, 1))
testData = np.hstack((testBias,testDatatemp))

outputtemp = np.genfromtxt(open("train.y.csv","rb"), dtype = 'str', delimiter=',',skip_header=1, usecols=(1))
output = np.zeros((NB_TRAININGEX,NB_CLASSES))

#initializing all the weigths randomly
syn1 = 2 * np.random.random((NB_FEATURES+1,NB_HIDDEN_NEURONS)) - 1
syn2 = 2 * np.random.random((NB_HIDDEN_NEURONS+1, NB_CLASSES)) - 1

#initializing the output matrix for the training data
j = 0
while j < NB_TRAININGEX:
	str = outputtemp[j]
	if str == 'International':
		output[j,0] = 1
	if str == 'Vocal':
		output[j,1] = 1
	if str == 'Latin':
		output[j,2] = 1
	if str == 'Blues':
		output[j,3] = 1
	if str == 'Country':
		output[j,4] = 1
	if str == 'Electronic':
		output[j,5] = 1
	if str == 'Folk':
		output[j,6] = 1
	if str == 'Jazz':
		output[j,7] = 1
	if str == 'New_Age':
		output[j,8] = 1
	if str == 'Pop_Rock':
		output[j,9] = 1
	if str == 'Rap':
		output[j,10] = 1
	if str == 'Reggae':
		output[j,11] = 1
	if str == 'RnB':
		output[j,12] = 1
	j = j+1

#values for reference
#international = 0
#vocal = 1
#latin = 2
#blues = 3
#country = 4
#electronic = 5
#folk = 6
#jazz= 7
#new-age=8
#pop_rock = 9
#rap = 10
#reggae = 11
#rnb = 12

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoidDeriv(x):
	return x *(1 - x)

#1 hidden layer = 2 synapses = 2 

def forwardPass (inputLayer, weights1, weigths2):
	hiddenLayer = inputLayer.dot(weights1)
	# apply sigmoid on all activations
	#no sigmoid on bias term so initialize i at 1
	i = 1 
	while i < NB_HIDDEN_NEURONS:
		hiddenLayer[i] = sigmoid(hiddenLayer[i])
		i = i + 1
	
	#add one for the bias term	
	hiddenLayer = np.hstack((1,hiddenLayer))
	result = hiddenLayer.dot(weigths2)

	# apply sigmoid on all activations

	#no sigmoid on bias term so initialize i at 1
	i = 1 
	while i < NB_CLASSES:
		result[i] = sigmoid(result[i])
		i = i + 1
		
	#normalize the data	
	i = 0
	sum = np.sum(result)
	while i < NB_CLASSES:
		result[i] = result[i]/sum
		i = i+1
	return result

result2 = forwardPass(inputFinal[0],syn1,syn2)


#calculate the error

def errorCalcul(desiredOutput, algorithmOutput, numberClasses):
	error = np.zeros(algorithmOutput.shape)
	for j in range(numberClasses):
		error[j]= desiredOutput[1,j]*np.log(algorithmOutput[j])
	return error

error2 = errorCalcul(output,result2,NB_CLASSES)


#testing
testOutput = np.zeros(NB_CLASSES+1)

i=0
while i < NB_TEST:
	resultat = forwardPass(testData[i],syn1,syn2)
	j = i+1
	resultat = np.hstack((j,resultat))
	testOutput = np.vstack((testOutput,resultat))
	i = i+1

testOutput = np.delete(testOutput, (0), axis=0)
testOutput.astype(np.int32)
#test data
with open('submission.csv','a') as f_handle:
	np.savetxt(f_handle, testOutput, fmt='%i,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f',delimiter=",")
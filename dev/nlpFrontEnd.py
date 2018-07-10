# -*- coding: utf-8 -*-

#%%
'''Import Modules'''
import customTokenizer
import machineLearningModule
import dataLayer
#%%
'''Get Data'''

dataLayer.importNlpTestData(100000)
testData = dataLayer.readTestData()

#%%
'''Run - Get vector of tokens'''
tokens = customTokenizer.customTokenizerVector(testData['reviewText'])

#%%   
'''Run - Get fit score for all methodologies'''

for i in [False, True]:
    for j in [False, True]:
        machineLearningModule.getMnbScore(x = testData['reviewText'], y = testData['overall'], TFID = i, customToken = j)

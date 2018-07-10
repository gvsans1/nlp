# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:13:42 2018

@author: gsansone
"""

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
 
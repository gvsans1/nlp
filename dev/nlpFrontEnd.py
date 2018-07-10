# -*- coding: utf-8 -*-

#%%
'''Import Modules'''
import pandas as pd
import statsmodels.api as sm
import customTokenizer
import machineLearningModule
import dataLayer
#%%
'''Get Data'''

dataLayer.importNlpTestData(10500)
testData = dataLayer.readTestData()

#%%
'''Run - Get vector of tokens (only as a test)'''
tokens = customTokenizer.customTokenizerVector(testData['reviewText'])

#%%   
'''Run - Get fit score for all methodologies'''

for i in [False, True]:
    for j in [False, True]:
        machineLearningModule.getMnbScore(x = testData['reviewText'], y = testData['overall'], TFID = i, customToken = j)

#%%   
'''Run - Obtain term-document matrix for best-fit model'''
'''Let's use CV with our custom tokenizer'''

cvCustom = machineLearningModule.getDocumentTermMatrix(x = testData['reviewText'], TFID = 0, customToken = 1)
#cvStd = machineLearningModule.getDocumentTermMatrix(x = testData['reviewText'], TFID = 0, customToken = 0)
#tfidCustom = machineLearningModule.getDocumentTermMatrix(x = testData['reviewText'], TFID = 1, customToken = 1)
#tfidStd = machineLearningModule.getDocumentTermMatrix(x = testData['reviewText'], TFID = 1, customToken = 0)
#%%   
'''Keep only columns (i.e. tokens) where the TFID sum value is above k-th percentile or k threshold'''

def reduceDocumentTermMatrix(chosenMatrix, method, k):
    sumScore = chosenMatrix.sum(axis = 0)
    if method == 'percentile':
        topTerms = sumScore[sumScore > sumScore.quantile(k)]
    elif method == 'max':
        topTerms = sumScore.nlargest(k)
    topTermsList = topTerms.index.tolist()
    out = chosenMatrix[topTermsList]    
    return out

dataTop = reduceDocumentTermMatrix(chosenMatrix = cvCustom, method = 'percentile', k = 0.999)
#dataTop = reduceDocumentTermMatrix(chosenMatrix = cvCustom, method = 'max', k = 10)

#%%
'''Create dataset for regression'''

regressionData = pd.merge(dataTop, pd.DataFrame(testData['overall']), how = 'left',left_index=True, right_index=True)

#%%
'''Fit Multinomial Logit'''

X = regressionData[dataTop.columns]
y = regressionData.overall
logitModel = sm.MNLogit(y,X)
result = logitModel.fit()
print(result.summary())
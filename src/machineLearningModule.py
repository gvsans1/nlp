# -*- coding: utf-8 -*-

"""
This module defines a function that obtains Naive Bayes Classifier Fit 
Scores. It compares train/test performance of different tokenizers (custom vs 
standard) and methodologies (CV vs TFID). This module requires the 
customTokenizer module to be imported.
"""

# %%

'''Import Libraries'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import customTokenizer

# %%

'''Get Naive Bayes Score'''


def getMnbScore(x, y, TFID, customToken):
    x = x.fillna(' ')  # Make NA in reviews blanks
    # x = x.values.astype('U') # Specify unicode encoding for reviews - 
    # switch on only in case of error - very slow
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.3,
                                                    random_state=3)
    nb = MultinomialNB()
    if TFID:
        if customToken:
            tfidf = TfidfVectorizer(
                tokenizer=customTokenizer.customTokenizerString)
            xTrainTfidf = tfidf.fit_transform(xTrain)
            xTestTfidf = tfidf.transform(xTest)  # Specify unicode encoding
            fit = nb.fit(xTrainTfidf, yTrain)
            score = nb.score(xTestTfidf, yTest)
            print("TFID with custom tokenizer: %f" % score)
            return score
            return fit
        else:
            tfidf = TfidfVectorizer()
            xTrainTfidf = tfidf.fit_transform(xTrain)
            xTestTfidf = tfidf.transform(xTest)  # Specify unicode encoding
            fit = nb.fit(xTrainTfidf, yTrain)
            score = nb.score(xTestTfidf, yTest)
            print("TFID with standard tokenizer: %f" % score)
            return score
            return fit
    else:
        if customToken:
            cv = CountVectorizer(
                tokenizer=customTokenizer.customTokenizerString)
            xTrainCV = cv.fit_transform(xTrain)
            xTestCV = cv.transform(xTest)  # Specify unicode encoding
            fit = nb.fit(xTrainCV, yTrain)
            score = nb.score(xTestCV, yTest)
            print("CV with custom tokenizer: %f" % score)
            return score
            return fit
        else:
            cv = CountVectorizer()
            xTrainCV = cv.fit_transform(xTrain)
            xTestCV = cv.transform(xTest)  # Specify unicode encoding
            fit = nb.fit(xTrainCV, yTrain)
            score = nb.score(xTestCV, yTest)
            print("CV with standard tokenizer: %f" % score)
            return score
            return fit


# %%
'''Create a document-term matrix with count vectorizer'''


# def getDocumentTermMatrixCV():

def getDocumentTermMatrix(x, TFID, customToken):
    x = x.fillna(' ')  # Make NA in reviews blanks
    # x = x.values.astype('U') # Specify unicode encoding for reviews - 
    # switch on only in case of error - very slow

    if TFID:
        if customToken:
            tfidf = TfidfVectorizer(
                tokenizer=customTokenizer.customTokenizerString)
            tokens = tfidf.fit_transform(x).toarray()
            matrix = pd.DataFrame(tokens, columns=tfidf.get_feature_names())
            return matrix
        else:
            tfidf = TfidfVectorizer()
            tokens = tfidf.fit_transform(x).toarray()
            matrix = pd.DataFrame(tokens, columns=tfidf.get_feature_names())
            return matrix
    else:
        if customToken:
            cv = CountVectorizer(
                tokenizer=customTokenizer.customTokenizerString)
            tokens = cv.fit_transform(x).toarray()
            matrix = pd.DataFrame(tokens, columns=cv.get_feature_names())
            return matrix
        else:
            cv = CountVectorizer()
            tokens = cv.fit_transform(x).toarray()
            matrix = pd.DataFrame(tokens, columns=cv.get_feature_names())
            return matrix


# %%


print('ML module successfully in place')

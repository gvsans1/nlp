"""Import Libraries"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from src.main.custom_tokenizer import custom_tokenizer_string

"""
This module defines a function that obtains Naive Bayes Classifier Fit
Scores. It compares train/test performance of different tokenizers (custom vs
standard) and methodologies (CV vs TFID). This module requires the
customTokenizer module to be imported.
"""

# %%

"""Get Naive Bayes Score"""


def get_mnb_score(x, y, tfid, custom_token):
    x = x.fillna(' ')  # Make NA in reviews blanks
    # x = x.values.astype('U') # Specify unicode encoding for reviews -
    # switch on only in case of error - very slow
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=.3, random_state=3)

    nb = MultinomialNB()

    if tfid:

        if custom_token:
            tfid = TfidfVectorizer(tokenizer=custom_tokenizer_string)
            x_train_tfid = tfid.fit_transform(x_train)
            x_test_tfid = tfid.transform(x_test)  # Specify unicode encoding
            fit = nb.fit(x_train_tfid, y_train)
            score = nb.score(x_test_tfid, y_test)
            print("TFID with custom tokenizer: %f" % score)
            return score, fit

        else:
            tfid = TfidfVectorizer()
            x_train_tfid = tfid.fit_transform(x_train)
            x_test_tfid = tfid.transform(x_test)  # Specify unicode encoding
            fit = nb.fit(x_train_tfid, y_train)
            score = nb.score(x_test_tfid, y_test)
            print("TFID with standard tokenizer: %f" % score)
            return score, fit

    else:

        if custom_token:
            cv = CountVectorizer(tokenizer=custom_tokenizer_string)
            x_train_cv = cv.fit_transform(x_train)
            x_test_cv = cv.transform(x_test)  # Specify unicode encoding
            fit = nb.fit(x_train_cv, y_train)
            score = nb.score(x_test_cv, y_test)
            print("CV with custom tokenizer: %f" % score)
            return score, fit

        else:
            cv = CountVectorizer()
            x_train_cv = cv.fit_transform(x_train)
            x_test_cv = cv.transform(x_test)  # Specify unicode encoding
            fit = nb.fit(x_train_cv, y_train)
            score = nb.score(x_test_cv, y_test)
            print("CV with standard tokenizer: %f" % score)
            return score, fit


# %%
"""Create a document-term matrix with count vectorizer"""


# def getDocumentTermMatrixCV():

def get_document_term_matrix(data, tfid, custom_token):
    data = data.fillna(' ')  # Make NA in reviews blanks
    # x = x.values.astype('U') # Specify unicode encoding for reviews -
    # switch on only in case of error - very slow

    if tfid:

        if custom_token:
            tfid = TfidfVectorizer(tokenizer=custom_tokenizer_string)
        else:
            tfid = TfidfVectorizer()

        tokens = tfid.fit_transform(data).toarray()
        matrix = pd.DataFrame(tokens, columns=tfid.get_feature_names())
        return matrix

    else:

        if custom_token:
            cv = CountVectorizer(tokenizer=custom_tokenizer_string)
        else:
            cv = CountVectorizer()

        tokens = cv.fit_transform(data).toarray()
        matrix = pd.DataFrame(tokens, columns=cv.get_feature_names())
        return matrix

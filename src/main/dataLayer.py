# -*- coding: utf-8 -*-


"""
This module reads from a gz file stored in data, containing information on 
Amazon Reviews.
The module first reads the gz file and extracts a number of reviews - 
currently 1000.
The module then reads and store the extracts in memory for use by the main.py.
"""

# %%
'''Import Test Data'''

import pandas as pd
import gzip


def importNlpTestData(numberOfReviews):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    pth = r"data\reviews_Musical_Instruments_5.json.gz"
    df = getDF(pth)

    df_test = df[df.index < numberOfReviews]

    df_test.to_csv("nlpTestData.csv", sep="\t")


# %%
'''Read Test Data'''


def readTestData():
    pth = r'data/nlpTestData.csv'
    testData = pd.read_csv(pth, sep="\t", engine='python')
    return testData


# %%
print("Data Layer succesfully in place")

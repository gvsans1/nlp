"""Import libraries"""
import pandas as pd
import gzip

"""
This module reads from a gz file stored in data, containing information on
Amazon Reviews.

    * The module first reads the gz file and extracts 
      a number of reviews.
    * The module then reads and store the extracts in memory
      for use by the main.py.
"""


def import_nlp_test_data(number_of_reviews):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def get_dataframe(path):
        i = 0
        dataframe = {}
        for d in parse(path):
            dataframe[i] = d
            i += 1
        return pd.DataFrame.from_dict(dataframe, orient='index')

    pth = r"data\reviews_Musical_Instruments_5.json.gz"
    df = get_dataframe(pth)

    df_test = df[df.index < number_of_reviews]

    df_test.to_csv("nlpTestData.csv", sep="\t")

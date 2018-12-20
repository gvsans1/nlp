"""
Import Modules
"""
import pandas as pd
import statsmodels.api as sm
from src.main.data_layer import import_nlp_test_data
from src.main.custom_tokenizer import custom_tokenizer_vector
from src.main.document_term_matrix_reduction import reduce_document_term_matrix
from src.main.machine_learning_module import (get_document_term_matrix,
                                              get_mnb_score)

# %%
"""
1. Get Test Data
"""

import_nlp_test_data(10500)
pth = r'data/nlpTestData.csv'
test_data = pd.read_csv(pth, sep="\t")

# %%
"""
2. Run - Get vector of tokens (only as a test)
"""
tokens = custom_tokenizer_vector(test_data['reviewText'])

# %%
"""
3. Run - Get fit score for all methodologies:
  * Frequency measure: Count vs TFID
  * Tokenizer: custom vs standard

"""

# Count Vectorizer and Standard Tokenizer
get_mnb_score(
    x=test_data['reviewText'],
    y=test_data['overall'], tfid=False,
    custom_token=False)

# TFID and Standard Tokenizer
get_mnb_score(
    x=test_data['reviewText'],
    y=test_data['overall'], tfid=True,
    custom_token=False)

# Count Vectorizer and Custom Tokenizer
get_mnb_score(
    x=test_data['reviewText'],
    y=test_data['overall'], tfid=False,
    custom_token=True)

# TFID and Custom Tokenizer
get_mnb_score(
    x=test_data['reviewText'],
    y=test_data['overall'], tfid=True,
    custom_token=True)

# %%
"""
4. Run - Obtain term-document matrix for best-fit model.
   * Use CV with our custom tokenizer
   * Reduce resulting document-term matrix filtering out low-importance tokens
"""

# Get document-term matrix using count vectorizer and our custom tokenizer
cv_custom = \
    get_document_term_matrix(
        data=test_data['reviewText'],
        tfid=False,
        custom_token=True)

# Keep only tokens above a certain percentile
dataTop = \
    reduce_document_term_matrix(
        chosen_matrix=cv_custom,
        method='percentile',
        k=0.999)

# %%
"""
5. Create dataset for regression
"""
regressionData = \
    pd.merge(
        dataTop,
        pd.DataFrame(test_data['overall']),
        how='left',
        left_index=True,
        right_index=True)

# %%
"""
6. Fit a Multinomial Logit
"""
x = regressionData[dataTop.columns]
y = regressionData.overall
logit_model = sm.MNLogit(y, x)
result = logit_model.fit()
print(result.summary())

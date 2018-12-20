"""
Function to keep only columns (i.e. tokens) where
the TFID sum value is above k-th percentile or k threshold
"""


def reduce_document_term_matrix(chosen_matrix, method, k):
    sum_score = chosen_matrix.sum(axis=0)
    if method == 'percentile':
        top_terms = sum_score[sum_score > sum_score.quantile(k)]
        top_terms_list = top_terms.index.tolist()
        out = chosen_matrix[top_terms_list]
    elif method == 'max':
        top_terms = sum_score.nlargest(k)
        top_terms_list = top_terms.index.tolist()
        out = chosen_matrix[top_terms_list]
    else:
        raise Exception('WARNING: reduce_document_term_matrix method '
                        'is invalid. Returning empty dataframe')
    return out

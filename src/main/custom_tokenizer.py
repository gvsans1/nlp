"""Import Libraries"""
# import nltk
# nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

# %%

""""
Custom Tokenizer module for Natural Language Processing

Process:

1. Prepare the data
    a. remove punctuation
    b. convert to lower case
    c. remove digits
    d. convert to tokens
    e. isolate the root of the tokenized term ("stemming")


2. Run a predictive model on it
    a. using a standard tokenizer
    b. using the custom tokenizer identified in point (1)

3. The predictive model can compute terms relevance by;
    a. count (count vectorizer)
    b. term frequency-inverse document frequency (TFID)


Terminology:

                           +------------------------+
                           |         TERMS          |
                           +------------------------+
                           | 'Ciao' | 'Marco' | '!' |
+-----------+--------------+--------+---------+-----+
|           |'Ciao Marco!' |    1   |    1    |  1  |
+ DOCUMENTS +--------------+--------+---------+-----+
|           |'[...]'       |    0   |    0    |  0  |
+-----------+--------------+--------+---------+-----+

"""

# %%


# %%

"""Define a custom tokenizer that works only on a single string"""


def custom_tokenizer_string(my_text):
    """Works only on plain string text, not vectorized"""

    # remove punctuation
    remove_punct = str.maketrans('', '', string.punctuation)
    text_no_punct = my_text.translate(remove_punct)

    # convert to lower case
    text_no_punct_lower = text_no_punct.lower()

    # remove digits
    remove_digits = str.maketrans('', '', string.digits)
    text_no_punct_lower_no_digits = text_no_punct_lower.translate(remove_digits)

    # tokenize
    tokenized_text = word_tokenize(text_no_punct_lower_no_digits)

    # remove stop words
    stop_words = stopwords.words('english')
    tokenized_text_no_stop = [y for y in tokenized_text if y not in stop_words]

    # stem
    stemmer = SnowballStemmer('english')
    tokenized_stems = [stemmer.stem(y) for y in tokenized_text_no_stop]

    return tokenized_stems


# %%
"""Test custom tokenizer - it accepts a simple plain text as required by 
sklearn predictive model"""
print('Custom Tokenizer successfully in place: ',
      custom_tokenizer_string('This is a test of my NLP Custom Tokenizer'))
# %%

# %%

"""Define a custom tokenizer that works on a vector of strings"""


def custom_tokenizer_vector(my_text):
    """Works on an array of strings"""

    # Make NA in reviews blanks
    my_text = my_text.fillna(' ')

    # remove punctuation
    remove_punct = str.maketrans('', '', string.punctuation)
    text_no_punct = my_text.str.translate(remove_punct)

    # convert to lower case
    text_no_punct_lower = text_no_punct.str.lower()

    # remove digits
    remove_digits = str.maketrans('', '', string.digits)
    text_no_punct_lower_no_digits = \
        text_no_punct_lower.str.translate(remove_digits)

    # tokenize
    tokenized_text = text_no_punct_lower_no_digits.map(word_tokenize)

    # remove stop words
    stop_words = stopwords.words('english')
    tokenized_text_no_stop = tokenized_text.map(
        lambda txt: [y for y in txt if y not in stop_words])

    # stem
    stemmer = SnowballStemmer('english')
    tokenized_stems = tokenized_text_no_stop.map(
        lambda txt: [stemmer.stem(y) for y in txt])

    return tokenized_stems

# -*- coding: utf-8 -*-

# %%

'''
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

'''

'''
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
    
'''

# %%
'''Import Libraries'''
# import nltk

# ONLY FIRST TIME
# nltk.set_proxy("https://del-webproxy.blackrock.com:8080")
# nltk.download('all')

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# %%

'''Define a custom tokenizer that works only on a single string'''


def customTokenizerString(myText):
    '''Works only on plain string text, not vectorized'''

    # remove punctuation
    removePunct = str.maketrans('', '', string.punctuation)
    textNoPunct = myText.translate(removePunct)

    # convert to lower case
    textNoPunctLower = textNoPunct.lower()

    # remove digits
    remove_digits = str.maketrans('', '', string.digits)
    textNoPunctLowerNoDigits = textNoPunctLower.translate(remove_digits)

    # tokenize
    tokenizedText = word_tokenize(textNoPunctLowerNoDigits)

    # remove stop words
    stopWords = stopwords.words('english')
    tokenizedTextNoStop = [y for y in tokenizedText if y not in stopWords]

    # stem
    stemmer = SnowballStemmer('english')
    tokenizedStems = [stemmer.stem(y) for y in tokenizedTextNoStop]

    return tokenizedStems


# %%
'''Test custom tokenizer - it accepts a simple plain text as required by 
sklearn predictive model'''
print('Custom Tokenizer successfully in place: ',
      customTokenizerString('This is a test of my NLP Custom Tokenizer'))
# %%

# %%

'''Define a custom tokenizer that works on a vector of strings'''


def customTokenizerVector(myText):
    '''Works only on plain string text, not vectorized'''

    # Make NA in reviews blanks
    myText = myText.fillna(' ')

    # remove punctuation
    removePunct = str.maketrans('', '', string.punctuation)
    textNoPunct = myText.str.translate(removePunct)

    # convert to lower case
    textNoPunctLower = textNoPunct.str.lower()

    # remove digits
    remove_digits = str.maketrans('', '', string.digits)
    textNoPunctLowerNoDigits = textNoPunctLower.str.translate(remove_digits)

    # tokenize
    tokenizedText = textNoPunctLowerNoDigits.map(word_tokenize)

    # remove stop words
    stopWords = stopwords.words('english')
    tokenizedTextNoStop = tokenizedText.map(
        lambda txt: [y for y in txt if y not in stopWords])

    # stem
    stemmer = SnowballStemmer('english')
    tokenizedStems = tokenizedTextNoStop.map(
        lambda txt: [stemmer.stem(y) for y in txt])

    return tokenizedStems

import pandas as pd

import vaderSentiment.vaderSentiment as vs


import sys
sys.path.append('../TextCleaner2000/text_cleaner_2000')
from text_cleaner import TextCleaner

def sentiment_analyzer_helper(doc):
    '''Parameters
    ----------
    doc: Text string. 
       
    Returns
    -------
    neg, neu, pos and compound sentiment score provided by Vader Sentiment.
    '''

    analyzer = vs.SentimentIntensityAnalyzer()

    try:

        s = analyzer.polarity_scores(doc)

    except:

        s = "{'neg': 0.00, 'neu': 0.00, 'pos': 0.00, 'compound': 0.00}"

    return s




import pandas as pd
import numpy as np 
import re

from pandas import DataFrame
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import pickle


def lemmatizer(text):
    '''Given text returns lemmatized text'''
    
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()
    
    return stemmer.stem(text)

def preprocess(document, stops, remove_numbers=False):
    '''Given a document return tokenized list that is lower-cased, stemmed and lemmatized'''
    
    results = [] 
    
    try:
        
        if remove_numbers:
            
            for text in document.split(' '):

                alpha_text = re.sub('[^a-zA-Z]+', ' ', text)

                for a in alpha_text.split(' '):
                    if a.lower() not in stops and a != ' ' and a != '':
                        results.append(a.lower())
        else:
            
            for text in document.split(' '):

                alpha_text = re.sub('[^a-zA-Z0-9]+', ' ', text)

                for a in alpha_text.split(' '):
                    
                    if a.lower() not in stops and a != ' ' and a != '':
                        
                        results.append(a.lower())
                       
    except:
        
        if remove_numbers:
            
            for text in str(document).split(' '):

                alpha_text = re.sub('[^a-zA-Z]+', ' ', text)

                for a in alpha_text.split(' '):
                    
                    if a.lower() not in stops and a != ' ' and a != '':
                        
                        results.append(a.lower())
                        
        else:
            
            for text in str(document).split(' '):

                alpha_text = re.sub('[^a-zA-Z0-9]+', ' ', text)

                for a in alpha_text.split(' '):
                    
                    if a.lower() not in stops and a != ' ' and a != '':

                        results.append(a.lower())
                       
    return results

def get_tokens(df):
    '''Given text returns a tokenized representation'''
    
    all_tokens = []
    
    for i in range(len(df)):
        
        for token in df['tokens'].iloc[i]:
            all_tokens.append(token)
            
    return all_tokens
            
def get_counts(df):
    '''Given token returns count of each token'''
    
    all_tokens = []
    
    for i in range(len(df)):
    
        for token in df['tokens'].iloc[i]:
            all_tokens.append(token)
        
    return pd.Series(all_tokens).value_counts()

def update_stops(stops):
    '''Given a list of stop words updates new_stops'''
    
    new_stops = []    
    
    for stop in stops:
        
        new_stops.append(stop)
        
    return new_stops

def update_tokens(text, new_stops):
    
    clean_tokens = []
    
    for t in text:
        
        if t not in new_stops:
            
            clean_tokens.append(t)
            
    return clean_tokens

def remove_html_tags(text):
    """Remove html tags from a string"""

    try:
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text).replace('\\n \\n', '')\
                                      .replace('\\r\\n', '')\
                                      .replace('\\r', '')\
                                      .replace('\\n', '')\
                                      .replace('  ', ' ')\
                                      .replace('&nbsp;','')\
                                      .replace('&rsquo','')\
                                      .lstrip()\
                                      .rstrip()
    except:
        return ' '

def lower_and_alpha(text):
    """Remove non-alphanumeric characters and lowercase text"""

    clean = re.compile('[^a-zA-Z0-9]')
    return re.sub(clean, ' ', text.lower()).replace('  ', ' ')
    



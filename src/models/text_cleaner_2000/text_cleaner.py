#!/usr/bin/env python
import re
import pickle
import os

import pandas as pd

class TextCleaner:
    """TextCleaner METHODS:
    TextCleaner.alpha_iterator():
    TextCleaner.alpha_iterator(text) ==>  Returns lower-case letters stripped of
    punctuation and numbers.
    By default alpha_iterator removes numbers and emoticons.To keep numbers:
        cleaner.alpha_iterator(text, remove_numeric = False)
        Likewise, to keep emoticons:
        cleaner.alpha_iterator(text, remove_emoticon = False)
        To Keep both:
        cleaner.alpha_iterator(text, remove_emoticon = False, remove_numeric = False)

    TextCleaner.stop_word_iterator():
    TextCleaner.stop_word_iterator(text) ==>  Removes common "stop" words, like "and".

    TextCleaner.custom_stop_word_iterator():
    TextCleaner.custom_stop_word_iterator(text, stop_words) ==> Custom stop_words in
    list format. stop_words are words to be removed.
    can use this in-lieu of stop_word_iterator, or in addition to. Alternatively, you can add words
    TextCleaner.stop_words list object, using TextCleaner.stop_words.append(word). This is the
    preferred method if you are using the in-built stopwords.

    TextCleaner.streaming_cleaner_and_tokenizer():
    TextCleaner.streaming_cleaner_and_tokenizer ==> A text cleaner, tokenizer, and stop-word remover
    that is designed to be used with bag-of-word, out-of-processor, or other algorithms
    that have an argument for a tokenizer.
    Method is applied for each row in a list. Pass this as an argument wherever you need a
    tokenizer/processor for algorithms like bag-of-words. Note that this method keeps emoticons,
    since they can help in determining polarity of text.

     STATIC METHODS:
        TextCleaner.tokenizer(text) ==> Returns tokens (unigrams) from a
        list of sentences.

    GENERAL USAGE:
    Import:
        2) from  TextCleaner2000.TextCleaner import TextCleaner
        Instance Instantiation:
        3) Simply  instantiate a cleaner object with empty call:
        cleaner = TextCleaner()

    CLASS METHOD USAGE:
        For the following examples, text refers to an array-like object.
        For best results, pass text as a list() or a Pandas
         DataFrame column: (assuming data_frame is a pandas DataFrame) data_frame["column_name"].
        For stop words used in custom stop word removal, pass stop words as a list().
    GENERAL NUMBER AND PUNCTUATION REMOVAL:
        alpha_words = cleaner.alpha_iterator(text, remove_emoticon = True, remove_numeric = True)
    COMMON STOPWORD REMOVAL:
        cleaned_of_stops = cleaner.stop_word_iterator(text)
    CUSTOM STOPWORD REMOVAL:
        cleaned_of_custom_stops = cleaner.custom_stop_word_iterator(text, stop_words)
        Remember that stop_words is a comma-separated list().
    STREAMING TEXT:
        Pass TextCleaner.streaming_cleaner_and_tokenizer as an argument to an algorithm
        that calls for a tokenizer. For example, with HashingVectorizer which can
        be used with SGDClassifier:
            HashingVectorizer(decode_error = 'ignore',
                        n_features = 2**21,
                        preprocessor = None,
                        tokenizer = cleaner.streaming_cleaner_and_tokenizer)"""

    def __init__(self):

        self.emoticons = None

        tc_2000_home = os.path.dirname(os.path.abspath(__file__))
        pickle_file_path = 'stops.pkl'

        pkl_file = open(os.path.join(tc_2000_home, pickle_file_path), 'rb')
        self.stop_words = pickle.load(pkl_file)

        pkl_file.close()

    @classmethod 
    def __alphaizer(self, text, remove_numeric, remove_emoticon):
        """Given a string (text), removes all punctuation and numbers.
        Returns lower-case words. Called by the iterator method
        alpha_iterator to apply this to lists, or array-like (pandas dataframe)
        objects."""

        if remove_numeric and remove_emoticon:
            non_numeric = ''.join(i for i in text if not i.isdigit())
            cleaned = re.sub('[^A-Za-z0-9]+', ' ', non_numeric)
            return cleaned.lower().strip()

        if not remove_numeric and remove_emoticon:
            cleaned = re.sub('[^A-Za-z0-9]+', ' ', text)
            return cleaned.lower().strip()

        if not remove_numeric and not remove_emoticon:

            emoticons = TextCleaner.__emoticon_finder(text)
            emoticons.replace('-', '')
            self.set_emoticons(emoticons)
            cleaned = re.sub('[^A-Za-z0-9]+', ' ', text)
            clean = cleaned.lower().strip() + ' ' + emoticons
            return clean

        if remove_numeric and not remove_emoticon:
            emoticons = TextCleaner.__emoticon_finder(text)
            emoticons.replace('-', '')
            self.set_emoticons(emoticons)
            
            non_numeric = ''.join(i for i in text if not i.isdigit())
            cleaned = re.sub('[^A-Za-z0-9]+', ' ', non_numeric)
            clean = cleaned.lower().strip() + ' ' + emoticons
            return clean

    @staticmethod
    def tokenizer(text):
        """Given a sentence, splits sentence on blanks and returns a list of ngrams or tokens"""
        if isinstance(text, str):
            tokenized = text.split(' ')
            clean = (token for token in tokenized if token != '')
            return list(clean)

        if isinstance(text, (pd.core.series.Series, list)):
            clean = (TextCleaner.tokenizer(t) for t in text if t is not None)
            return list(clean)

    @staticmethod
    def __stop_word_remover(text, stop):
        """Removes common stop-words like: "and", "or","but", etc. Called by
        stop_word_iterator to apply this to lists, or array-like (pandas dataframe)
        objects. """

        clean = ''
        tokens = (t for t in TextCleaner.tokenizer(text) if t not in stop)
        for t in list(tokens):
            clean += " " + t
        return clean.lstrip()

    @staticmethod
    def __emoticon_finder(text):
        """Finds emoticons."""

        emoticons_ = ""
        emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        for e in emoticons:
            emoticons_ += ' ' + e
        return emoticons_.lstrip()

    #@classmethod 
    def stop_word_iterator(self, text):
        """Calls __stop_word_remover to apply this method to array-like objects.
        Usage: TextCleaner.stop_word_iterator(Text)."""

        clean = (self.__stop_word_remover(t, self.stop_words) for t in text)
        return list(clean)

    @classmethod 
    def alpha_iterator(self, text, remove_numeric=True, remove_emoticon=True):
        """Calls __alphaizer to apply this method to array-like objects. Usage:
        TextCleaner.alphaizer(Text).
        Note: By default this method removes numbers from each string.
        To change this behavior pass the flag remove_numerals:
        alphaizer(Text, remove_numerals = False)
        """

        clean = (self.__alphaizer(t, remove_numeric, remove_emoticon) for t in text)
        return list(clean)
    
    @classmethod 
    def custom_stop_word_iterator(self, text, stop_words):
        """Removes custom stop-words. For cleaned example, "patient", or "medicine", if
        one is dealing with medical Text and do not want to include those words
        in analysis. Can use this method to pass any set of stop
        words, or in-lieu of common stop-word method stop_word_iterator.Calls
        __stop_word_remover to apply this method to array-like objects. Usage:
        TextCleaner.custom_stop_word_iterator(Text, stop_words), where
        stop-words and Text are in a comma-
        separated list, or iterable."""

        clean = (self.__stop_word_remover(t, stop_words) for t in text)
        return list(clean)
    
    @classmethod 
    def streaming_cleaner_and_tokenizer(self, text, remove_numeric=True, remove_emoticon=False):
        """Called per line of text in a a stream application, such as SGD.
        Can be passed directly to SGD algorithms as TextCleaner.streaming_cleaner_and_tokenizer"""

        alpha = self.__alphaizer(text=text, remove_numeric=remove_numeric,
                                 remove_emoticon=remove_emoticon)

        clean = self.__stop_word_remover(alpha, self.stop_words)
        tokens = TextCleaner.tokenizer(clean)
        return tokens
    
    @classmethod 
    def set_emoticons(self, emoticons):
        """Called from __alphaizer. 
        Sets self.emoticons value to be retrieved."""

        self.emoticons = emoticons


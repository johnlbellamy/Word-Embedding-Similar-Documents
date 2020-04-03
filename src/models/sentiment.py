#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from ast import literal_eval

import vaderSentiment.vaderSentiment as vs

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '', 'text_cleaner_2000'))
from text_cleaner_2000.text_cleaner import TextCleaner

from sentiment.vader_helper import *

from tqdm import tqdm

if sys.argv[1] == 'svd':

    df = pd.read_csv('data/processed/data_svd_tagged.csv')

if sys.argv[1] == 'embed':

    df = pd.read_csv('data/processed/data_embed_tagged.csv')

print("Removing non-alphas...")
print()

cleaner = TextCleaner()

alpha_tokens = cleaner.alpha_iterator(df['par_text_clean'])

print("Removing common stop words...")
print()

clean_tokens = cleaner.stop_word_iterator(alpha_tokens)

print("Getting sentiment scores...")
print()

with tqdm(total=len(clean_tokens)) as pbar:

    sentiment_scores = []

    for token in clean_tokens:
        sentiment_scores.append(sentiment_analyzer_helper(token))

        pbar.update(1)

print("Saving results to file...")

sentiment_frame = pd.DataFrame(sentiment_scores)

df_join = df.join(sentiment_frame)

if sys.argv[1] == 'svd':

    df_join.to_csv('data/processed/svd_data_tagged_and_sentiment.csv', index=False)

if sys.argv[1] == 'embed':

    df_join.to_csv('data/processed/embed_data_tagged_and_sentiment.csv', index=False)

print('Done!')
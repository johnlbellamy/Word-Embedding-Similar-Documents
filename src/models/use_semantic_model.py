#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from ast import literal_eval

import tensorflow as tf
import tensorflow_hub as hub

import re
import sys

from use_model.use import *

from tqdm import tqdm

print("Load Data...")
print()

df = pd.read_csv('data/interim/jobs6_clean_pars.csv')

df['par_text_clean'] = [x.replace(']', '').replace(
    '[', '').replace(',', '').replace("'", '') for x in df['par_text_clean']]

df = df.drop_duplicates('par_text_clean')

df['tokens'] = [x.split(' ') for x in df['par_text_clean']]

indexes = []

for index, row in df.iterrows():

    if len(row['tokens']) > 1:

        indexes.append(index)

df = df.loc[indexes]

df = df.iloc[0:20000]

df = df.reset_index()

print("Prepare USE...")
print()

embed = hub.Module('src/models/use')

chunks = list(chunker(df['par_text_clean'], 500))  # Chunks to avoid OOM errors

print('Run USE... Will tak some time.\nIterations are pretinted out. Expect 40 iterations.')
print()

results = get_features_iterator(chunks, embed)

similar_docs = []
error_indexes = []

print('Find similar documents for each row...')
print()

with tqdm(total=len(df)) as pbar:

    try:
        for index, row in df.iterrows():

            search = semantic_search(
                results[index], df['par_text_clean'], results)

            for s in search[1:6]:

                similar_docs.append([index, s[1], s[0], s[2]])

            pbar.update(1)

    except IndexError:

        error_indexes.append(index)
        pbar.update(1)

print('Prepare results for saving...')
print()

index = [x[0] for x in similar_docs]
similarity = [x[2] for x in similar_docs]
ids = [x[3] for x in similar_docs]

s = pd.DataFrame.from_records([similarity, ids, index]).T

s.columns = ['max_support', 'id', 'index']

s['id'] = s['id'].astype(int)
s['index'] = s['index'].astype(int)

result_dict = {}

for i in s['index']:

    result_dict[i] = list(s[s['index'] == i]['id'])

df = df.loc[result_dict.keys()]

job_dict = {}
index = 0

for job in df['Title'].unique():

    job_dict[job] = index
    index += 1

df['job_no'] = df['Title'].map(lambda x: job_dict[x])

df['chunk_no'] = df.index

df['similar_documents'] = df.index.map(lambda x: result_dict[x])

df = df[['Title', 'par_text', 'par_text_clean',
         'par', 'similar_documents', 'job_no', 'chunk_no']]

print('Saving to file...')
print()
df.to_csv('data/processed/data_embed_tagged.csv', index=False)

print("Done!")


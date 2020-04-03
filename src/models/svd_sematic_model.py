#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from ast import literal_eval


from svd.semantic_model import *

from tqdm import tqdm

print('Loading data...')
print()

df = pd.read_csv('data/interim/jobs6_clean_pars.csv')

df = df.drop_duplicates('par_text_clean')

print('Creating tf-idf matrix.')
print()

tfidf_matrix = get_tfidf(df['par_text_clean'])

print('Make tf-idf scipy.parse')
print()

csc = scipy.sparse.csc_matrix(tfidf_matrix)
u, s, vt = get_sparse_objects(csc, 50)
u_prime = get_uprime(u, s)

print('Finding similar documents...')
print()

similar_doc_indexes = []
max_supports = []

with tqdm(total=len(df)) as pbar:
    
    for index, row in df.iterrows():

        blurred_field = get_blurred_field(u_prime, vt, index)
        max_support = max(blurred_field)

        token_ids = get_token_ids(blurred_field, max(blurred_field))

        if max_support <= 0.00 or len(token_ids[0].tolist()) == 0:

            similar_doc_indexes.append([-999])
            max_supports.append(max_support)

        else:

            similar_doc_indexes.append(token_ids[0].tolist())
            max_supports.append(max_support)
        pbar.update(1)

print('Done!')
print()

print('Saving results...')

df['similar_documents'] = similar_doc_indexes 

df['max_support'] =  max_supports

df.head()

df['job_no'] = df.index

df = df[['Title','par_text', 'par_text_clean', 'par', 'similar_documents', 'max_support', 'job_no']]

df.to_csv('data/processed/data_svd_tagged.csv', index=False)

print('Done! Go on to step 4) make embedded')


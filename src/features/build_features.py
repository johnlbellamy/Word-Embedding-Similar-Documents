#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import pickle
import sys, os

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'data_prep'))
from data_prepper.data_prepper import *


print("Loading data into dataframe...")
print()

df = pd.read_csv('data/raw/jobs6.tsv', sep='\t')

print("Begin cleaning and transforming...")
print()


print("Drop duplicates and take sample of data for computing limitations...")
print()

df = df.drop_duplicates('Description')

df = df.drop_duplicates('Requirements')

df = df.sample(n=10000, replace=True, random_state=0)

print("Combine requirements and description...")
print()

desc_and_req = []

with tqdm(total=len(df)) as pbar:
    for _, row in df.iterrows():

        try:
            desc_and_req.append(row['Description'] + ' ' + row['Requirements'])
            pbar.update(1)

        except TypeError:  # always with the floats!

            try:
                desc_and_req.append(
                    str(row['Description']) + ' ' + row['Requirements'])
                pbar.update(1)

            except TypeError:

                try:

                    desc_and_req.append(
                        row['Description'] + ' ' + str(row['Requirements']))
                    pbar.update(1)

                except TypeError:

                    desc_and_req.append(
                        str(row['Description']) + ' ' + str(row['Requirements']))
                    pbar.update(1)


df['desc_and_req'] = desc_and_req

print()
print("Split into paragraphs...")
print()

text_and_par = []

with tqdm(total=len(df) * 7) as pbar:
    for _, row in df.iterrows():

        try:
            par_no = 1

            for par_text in row['desc_and_req'].split('</p>'):
                if par_text != '' and par_text != ' ':
                    text_and_par.append([row['Title'], par_text, par_no])
                par_no = par_no + 1
                pbar.update(1)

        except AttributeError:  # always with the floats!

            if par_text != '' and par_text != ' ':
                text_and_par.append(
                    [row['Title'], row['desc_and_req'], par_no])
            par_no = par_no + 1
            pbar.update(1)

text_and_par_df = pd.DataFrame(
    text_and_par, columns=['Title', 'desc_and_req', 'par'])

print()
print("Begin cleaning...")
print()

text_and_par_df['par_text'] = text_and_par_df['desc_and_req'].map(lambda x: x.replace(
    'Please refer to the Job Description to view the requirements for this job', ''))

print()
print("Remove html tags...")
print()

par_text = []

with tqdm(total=len(text_and_par_df)) as pbar:
    for text in text_and_par_df['par_text']:
        par_text.append(remove_html_tags(text))
        pbar.update(1)

text_and_par_df['par_text'] = par_text

print()
print("Lower and return only alpha-numeric...")
print()

par_text = []

with tqdm(total=len(text_and_par_df)) as pbar:
    for text in text_and_par_df['par_text']:
        par_text.append(lower_and_alpha(text))
        pbar.update(1)

text_and_par_df['par_text_clean'] = par_text

text_and_par_df['par_text_clean'] = text_and_par_df['par_text_clean']\
                                    .map(lambda x: x.replace('job requirements', ''))

stops = pickle.load(open('data/stops.pkl', 'rb'))

stops.append('requirements')
stops.append('requirement')
stops.append('description')
stops.append('jobs')
stops.append('job')

print()
print("Removing stopwords...")
print()

par_text_clean = []

with tqdm(total=len(text_and_par_df)) as pbar:
    for text in text_and_par_df['par_text_clean']:
        par_text_clean.append(preprocess(text, stops))
        pbar.update(1)


text_and_par_df['par_text_clean'] = par_text_clean

print()
print("Remove any empty token lists...")
print()

text_and_par_df = text_and_par_df[text_and_par_df.astype(str)[
    'par_text_clean'] != '[]']

print()
print("Done... Saving file for next step...")
print()

text_and_par_df[['Title', 'par_text', 'par_text_clean', 'par']].to_csv(
    'data/interim/jobs6_clean_pars.csv')
    
print()
print("Done... Go on to step 3) make use.")
print()

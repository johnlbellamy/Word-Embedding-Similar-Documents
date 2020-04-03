import scipy
from scipy.sparse.linalg import svds
from sparsesvd import sparsesvd

from collections import defaultdict
from string import Template

#Machine  Learningconda isntall numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np 

def get_sparse_objects(csc, k, component='all'):
    '''Parameters
       ----------
       csc: compressed sparse column matrix (scipy.sparse.csc_matrix) 
       k: singular values requested.
       
       Returns
       -------
       component, where component is:
           u: Unitary Matrix left singular vectors,
           s: Singular values where shape=k,
           vt: Unitary Matrix right singular vectors
           us: u and s for collaborative filter.
           all(default): u, s and vt.
    '''
    u, s, vt = sparsesvd(csc, k)
    
    if component == 'u':
        return u
    
    elif component == 's':
        return s
    
    elif component == 'vt':
        return vt
    
    elif component == 'us':
        
        return u, s
    
    elif component == 'all':
        
        return u, s, vt
    
    
def get_uprime(u, s):
    '''Parameters
       ----------
       u: Unitary Matrix of left singular vectors and singular values (see get-sparse_objects).
       s: Singular values where shape=k (see get_sparse_objects).
       
       Returns
       -------
       matrix of dot product of u.T and the diagonal of s
    '''
    
    return np.dot(u.T, np.diag(s))

def get_blurred_field(u_prime, vt, index):
    '''Parameters
       ----------
       uprime: dot product of u.T and the diagonal of s and vt (unitary matrix of right values). See: get_sparse_objects.
       Returns
       -------
       dot product of u_prime and vt.
    '''
    
    return np.dot(u_prime, vt[:,index])

def get_token_ids(blurred_field, support):
    '''Given a blurred_field and support returns blurred_field which is >= support.
    
       Parameters
       ----------
       blurred_field: dot product of u_prime and vt based on an index. See: get_blurred_field.
       
       Returns
       -------
       blurred_fields: filtered blurred_fields that are greater or equal to given support.
    '''
    
    return np.where(blurred_field >= support)
        

def get_counts_and_doc_num(sources): 
    '''Parameters
       ----------
       source: tokens (comma-delimited list of stemmed-words) from a document such as a .docx page.
       
       Returns
       -------
       output: list of (doc_num, {term, num_occurences})
    '''
    
    index = 0
    output = []
    counts = {}
    
    for source in sources:
        counts = {}
        
        for s in source.split(','):
            
            if s.replace("'",'').replace('[', '').replace(']','').lstrip() in counts:
                counts[s.replace("'",'').replace('[', '').replace(']','').lstrip()] += 1
            
            else:
                counts[s.replace("'",'').replace('[', '').replace(']','').lstrip()] = 1
                
        output.append((index, counts))
        index += 1
    
    return output
   
def find_maximum_support(blurred_field):
    '''Given a blurred_field (see get_blurred_field) returns maximum support for a given similar document. 
       This is the parameter for get_token_ids.
    
       Parameters
       ----------
       
       blurred_field: (filtered matrix/dot product of u_prime and vt) returns maximum support (similarity) available. Helper
       method for assign_maximum_support.
       
       Returns
       -------
       suggested_support: float 3 places represented maximum similarity to another document in corpus.
    '''
    
    suggested_support = .96
    
    while len(get_token_ids(blurred_field, suggested_support)[0]) == 0 and suggested_support >= 0.0:
        
        suggested_support -= 0.00025
        
    if suggested_support >= 0.0:
        return abs(round(suggested_support, 3))
    else:
        return 0.00
        
def assign_maximum_support(u_prime, vt):
    '''Given u_prime and vt, returns the maximum support for a given similar document. This is the parameter
       for get_token_ids.
       
       Parameters
       ----------
       u_prime: Unitary Matrix of left singular vectors and singular values. See: get_sparse_objects.
       vt: Unitary Matrix right singular vectors. See: get_sparse_objects
       
       Returns
       -------
       maximum_supports: one-dimensional list with length n == len(u_prime)
    '''
    
    maximum_supports = []
    
    for i in range(len(u_prime)):
 
        blurred_field = get_blurred_field(u_prime, vt, i)
        maximum_supports.append(find_maximum_support(blurred_field))
  
    return maximum_supports

def get_tfidf(data):
    '''Given data, where data is a list
    '''
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=250000,
                                 min_df=2, stop_words=None,
                                 use_idf=True, ngram_range=(1,3), tokenizer=None)

    tfidf_matrix = tfidf_vectorizer.fit_transform(data) #fit the vectorizer to synopses

    return tfidf_matrix


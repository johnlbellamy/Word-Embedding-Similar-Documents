
import numpy as np 

import tensorflow as tf
import tensorflow_hub as hub

def get_features(texts):
    
    if type(texts) is str:
        
        texts = [texts]
        
    with tf.Session() as sess:
        
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        
        return sess.run(embed(texts))

def get_features_iterator(texts, embed):
    errors = []
    index = 1
    
    results = np.empty((0, 512), dtype=float)
    
    
    with tf.Session() as sess:
        
        
        try: 
          
            for text in texts:
               
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                
                print(index)
                res = sess.run(embed(text))
        
                results = np.vstack((results, res))
            
                index += 1
            
            return results
        
        except:
            
            print(index)
            
            index += 1
    
def cosine_similarity(v1, v2):
    
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if (not mag1) or (not mag2):
        
        return 0
    
    return np.dot(v1, v2) / (mag1 * mag2)

def test_similarity(text1, text2):
    
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    
    return cosine_similarity(vec1, vec2)

def semantic_search(query_vec, data, vectors):
    
    res = []
    
    for i, d in enumerate(data):
        
        qvec = vectors[i].ravel()
        sim = cosine_similarity(query_vec, qvec)
        res.append((sim, d[:100], i))
        
    return sorted(res, key=lambda x : x[0], reverse=True)

def chunker(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


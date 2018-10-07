# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

M = np.asarray([[3,7,4,9,9,7], 
                [7,0,5,3,8,8],
                [7,5,5,0,8,4],
                [5,6,8,5,9,8],
                [5,8,8,8,10,9],
                [7,7,0,4,7,8]])


M = pd.DataFrame(M) 


global k, metric

k=4 # k similar users
metric = 'cosine' #similarity metric, Pearson similarity can be used too

cosine_sim = 1-pairwise_distances(M, metric=metric)

user_id = 3


def find_k_similar_users(user_id,ratings,metric = metric, k = k):
    
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1].values.reshape(1, -1), n_neighbors = k+1)
    
    similarities = 1-distances.flatten()
    
    print('{0} most similar users for User {1}:\n'.format(k,user_id))
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;
    
        else:
            print('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i]+1, similarities.flatten()[i]))
            
    
    return similarities,indices

similarities,indices = find_k_similar_users(user_id,M,metric,k)



def predict_userbased(user_id,item_id, ratings, metric = metric, k = k):
    
    prediction = 0
    
    similarities,indices = find_k_similar_users(user_id,ratings,metric,k)
    
    
    mean_rating = ratings.loc[user_id-1].mean()
    
    sum_wt = np.sum(similarities) - 1
    
    product = 1
    
    weighted_sum = 0
    
    
    iflatten = indices.flatten()
    
    for i in range(0,len(iflatten)):
        
        if((i+1) == user_id):
            continue
        
        else:
            
            ratings_diff = ratings.iloc[iflatten[i],item_id - 1] - np.mean(ratings.iloc[iflatten[i]])
            
            product = ratings_diff * (similarities[i])
            
            weighted_sum += product
            
        prediction = int(round(mean_rating + (weighted_sum/sum_wt))) 
        
        
    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))
    
    return prediction

predict_userbased(3,4,M)
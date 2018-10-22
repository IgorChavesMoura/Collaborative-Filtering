# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import warnings

from sklearn.neural_network import MLPClassifier

#M = np.asarray([[3,7,4,9,9,7], 
#                [7,0,5,3,8,8],
#                [7,5,5,0,8,4],
#                [5,6,8,5,9,8],
#                [5,8,8,8,10,9],
#                [7,7,0,4,7,8]])


#M = np.random.randint(low=0,high=10,size=(100,6))

M = pd.read_csv('dataset.csv',header=None)

selected_item = 2
selected_user = 8

m_train = M.loc[lambda df:df[selected_item] != 0,:]

m_test = M.loc[lambda df:df[selected_item] == 0,:]



train_target_values = m_train[selected_item].values

train_featured_values = m_train.drop(m_train.columns[selected_item], axis=1).values

test_target_values = m_test[selected_item].values

test_featured_values = m_test.drop(m_test.columns[selected_item], axis=1).values

neural_network = MLPClassifier(verbose=True,max_iter=2000,tol=0.00001)

user_row = M.iloc[selected_user].drop(M.columns[selected_item]).values.reshape(1,-1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    neural_network.fit(train_featured_values,train_target_values)
    result = neural_network.predict(user_row)

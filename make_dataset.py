# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np




M = np.random.randint(low=0,high=10,size=(100,6))

M = pd.DataFrame(M) 

M.to_csv('dataset.csv',index=False)
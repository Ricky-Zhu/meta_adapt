import numpy as np
res = [[0.5 , 0.7 , 0.8 , 0.72 ,0.84],
       [0.42 ,0.68 ,0.68 ,0.4  ,0.58],
       [0.5,  0.58 ,0.7 , 0.56 ,0.52]]
res = np.array(res)
mean = np.mean(res,axis=0)
std = np.std(res,axis=0)
print(mean,std)
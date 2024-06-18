import pickle as pkl
import numpy as np


with open('test.pkl','rb') as f:
    data = pkl.load(f)

print(data)

import pickle 
import numpy as np
import matplotlib as plt

results = []
with open('results/results.pkl', 'rb') as f:
    results = pickle.load(f)

for _ in results:
    print(_['learning_params'])
    print(len(_['train_loss']['epoch']))

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import os
from tqdm import tqdm
from scipy.interpolate import griddata
from pyinstrument import Profiler

dataset_count = 19        
dataset_prefix = '../data/'
event_count = np.zeros(dataset_count, dtype=int)
for data_id in range(dataset_count):
    with h5py.File(f'{dataset_prefix}{16930+data_id}.h5', 'r') as data_file: 
        event_count[data_id] = data_file['ParticleTruth'].shape[0]

event_total = event_count.sum()
    
def get_data():
    # Read the geometry of detector
    with h5py.File(f'{dataset_prefix}geo.h5', 'r') as geo_file:
        ChannelID_ = geo_file['Geometry']['ChannelID']
        theta_ = geo_file['Geometry']['theta']
        phi_ = geo_file['Geometry']['phi'] - 180 # let phi in [-180, 180]
        geo_dict = {key: (val1, val2) for key, val1, val2 in zip(ChannelID_, theta_, phi_)}
        
    # Count events in datasets
    X = np.zeros(event_total)
    Y = np.zeros(event_total)
    Z = np.zeros(event_total)
    PE_total_train = np.zeros(event_total)
    
    # vecotrize the lookup function of geo_dict
    event_index = np.insert(np.cumsum(event_count), 0, 0)
    vectorized_lookup = np.vectorize(geo_dict.get)

    for data_id in tqdm(range(dataset_count)): # tqdm把iterator包起来，就可以实现进度条
        with h5py.File(f'{dataset_prefix}{16930+data_id}.h5', 'r') as data_file:
            
            X[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['x'][...]
            Y[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['y'][...]
            Z[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['z'][...]
            EventIDs = data_file['PETruth']['EventID'][...]
            _, PE_total = np.unique(EventIDs, return_counts=True) 
            PE_total_train[event_index[data_id]:event_index[data_id+1]] = PE_total
    
    pos = np.column_stack((X, Y, Z))
    
    with open(f'{dataset_prefix}scatters.pkl', 'rb') as f:
        Ek_train, Evis_train, pos, PE_total_train, ArrivalTime, ArrivalCount = pickle.load(f)
        
    dataset = (Ek_train, Evis_train, pos, PE_total_train, ArrivalTime, ArrivalCount)
    with open(f'{dataset_prefix}scatters.pkl', 'wb') as f:
        pickle.dump(dataset, f)
        
get_data()
import torch
import pickle
import numpy as np
import h5py
from utils import *
from deeponet import DeepONet

device = 'cuda:6' if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else 'cpu')

def train(x_train, y_train, x_test, y_test):
    model = DeepONet(branch=[3, 128, 128, 128, 128],
                    trunk_layers=[1, 128, 128, 128])
    # model = DeepONet(branch=Simple1DCNN(),
    #                  trunk_layers=[1, 128, 128, 128])

    model.run(x_train, y_train, x_test, y_test, batch_size=500, device=device, criterion='l2')
    model.predict(x_test, device=device)
    model.save('../model/')

def simulate(model, x_normalizer, y_normalizer, z_normalizer, PE_normalizer):
    if isinstance(model, str):
        model = torch.load(model)
    
    with open('../data/problem.pkl', 'rb') as f:
        x_problem, y_problem, z_problem, PE_total_problem, _, _ = pickle.load(f)
        PE_total_problem = PE_total_problem.astype('float64')
        x_problem = x_normalizer.encode(torch.from_numpy(x_problem[:, np.newaxis]).float())
        y_problem = y_normalizer.encode(torch.from_numpy(y_problem[:, np.newaxis]).float())
        z_problem = z_normalizer.encode(torch.from_numpy(z_problem[:, np.newaxis]).float())
        PE_total_problem = PE_normalizer.encode(torch.from_numpy(PE_total_problem[:, np.newaxis, np.newaxis]).float())
    
    x_problem = (torch.cat([x_problem, y_problem, z_problem], dim=1), PE_total_problem)
    Evis_problem = model.predict(x_problem, device=device).reshape(-1)
    Ek_problem = Evis_problem - 2 * 0.511
    EventID_problem = np.arange(1, 10001)

    ans_dtype = np.dtype([
            ('EventID', '<i4'),
            ('Ek', '<f4'),
            ('Evis', '<f4')
        ])
    ans_data = np.zeros(Ek_problem.shape, dtype=ans_dtype)
    ans_data['EventID'], ans_data['Ek'], ans_data['Evis'] = EventID_problem, Ek_problem, Evis_problem

    with h5py.File(f'ans.h5', 'w') as answer_file:
        answer_file.create_dataset('Answer', data=ans_data)
        
if __name__ == '__main__':
    with open('../data/data_train.pkl', 'rb') as f:
        x_train, y_train, z_train, PE_total_train, Ek_train, Evis_train = pickle.load(f)
        x_normalizer = Normalizer(x_train)
        y_normalizer = Normalizer(y_train)
        z_normalizer = Normalizer(z_train)
        PE_total_train = PE_total_train.astype('float64')
        PE_normalizer = Normalizer(PE_total_train)
        
        x_train = x_normalizer.encode(torch.from_numpy(x_train[:, np.newaxis]).float())
        y_train = y_normalizer.encode(torch.from_numpy(y_train[:, np.newaxis]).float())
        z_train = z_normalizer.encode(torch.from_numpy(z_train[:, np.newaxis]).float())
        PE_total_train = PE_normalizer.encode(torch.from_numpy(PE_total_train[:, np.newaxis, np.newaxis]).float())
        
    with open('../data/data_test.pkl', 'rb') as f:
        x_test, y_test, z_test, PE_total_test, Ek_test, Evis_test = pickle.load(f)
        PE_total_test = PE_total_test.astype('float64')
        x_test = x_normalizer.encode(torch.from_numpy(x_test[:, np.newaxis]).float())
        y_test = y_normalizer.encode(torch.from_numpy(y_test[:, np.newaxis]).float())
        z_test = z_normalizer.encode(torch.from_numpy(z_test[:, np.newaxis]).float())
        PE_total_test = PE_normalizer.encode(torch.from_numpy(PE_total_test[:, np.newaxis, np.newaxis]).float())

    x_train = (torch.cat([x_train, y_train, z_train], dim=1), PE_total_train)
    x_test = (torch.cat([x_test, y_test, z_test], dim=1), PE_total_test)
    y_train = torch.from_numpy(Evis_train[:, np.newaxis]).float()
    y_test = torch.from_numpy(Evis_test[:, np.newaxis]).float()
    
    # train(x_train, y_train, x_test, y_test)
    simulate('../model/2023-11-03_17-33-39.pth', x_normalizer, y_normalizer, z_normalizer, PE_normalizer)
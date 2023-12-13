import torch
import pickle
import numpy as np
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from model import *

device = 'cuda' if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else 'cpu')

def train(x_train, y_train, x_test=None, y_test=None, it=50, batch_size=1000):
    model = DeepONet(out_features=128, trunk_layers=[1, 128, 128, 128]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # select first 10000 samples as normalizer
    x_branch_normalizer = Normalizer(x_train[0][:10000])
    x_trunk_normalizer = Normalizer(x_train[1][:10000])
    # y_normalizer = Normalizer(y_train[:10000])
    
    # 创建 DataLoader
    train_data = TensorDataset(x_train[0], x_train[1], y_train) # branch input, trunk input, label
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(it):
        for i, (branch_input, trunk_input, labels) in enumerate(train_loader):
            branch_input, trunk_input, labels = branch_input.to(device), trunk_input.to(device), labels.to(device)
            branch_input = x_branch_normalizer.encode(branch_input)
            trunk_input = x_trunk_normalizer.encode(trunk_input)
            # labels = y_normalizer.encode(labels)
            
            optimizer.zero_grad() 

            outputs = model(branch_input, trunk_input)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:  
                print(f'Epoch [{epoch+1}/{it}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                print(f'Score:{1000 * myloss(outputs, labels):.4f}')

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
    with open('../data/eventimage.pkl', 'rb') as f:
        EventImage = pickle.load(f)
        ArrivalTime, ArrivalCount = zip(*EventImage)
        ArrivalTimeImage = torch.from_numpy(np.stack(list(ArrivalTime)))
        ArrivalCountImage = torch.from_numpy(np.stack(list(ArrivalCount)))
    
    with open('../data/scatters.pkl', 'rb') as f:
        Ek_train, Evis_train, pos, PE_total_train, ArrivalTime, ArrivalCount = pickle.load(f)
    
    x_branch_train = torch.cat([ArrivalTimeImage.unsqueeze(1), ArrivalCountImage.unsqueeze(1)], dim=1)
    x_trunk_train = torch.from_numpy(PE_total_train).unsqueeze(-1).unsqueeze(-1)
    y_train = torch.from_numpy(Ek_train).unsqueeze(-1)
    
    model = train((x_branch_train.to(torch.float32), x_trunk_train.to(torch.float32)), 
                  y_train.to(torch.float32), 50, 1000)
    
        # z_train = z_normalizer.encode(torch.from_numpy(z_train[:, np.newaxis]).float())
        # PE_total_train = PE_normalizer.encode(torch.from_numpy(PE_total_train[:, np.newaxis, np.newaxis]).float())
        
    # with open('../data/data_test.pkl', 'rb') as f:
    #     x_test, y_test, z_test, PE_total_test, Ek_test, Evis_test = pickle.load(f)
    #     PE_total_test = PE_total_test.astype('float64')
    #     x_test = x_normalizer.encode(torch.from_numpy(x_test[:, np.newaxis]).float())
    #     y_test = y_normalizer.encode(torch.from_numpy(y_test[:, np.newaxis]).float())
    #     z_test = z_normalizer.encode(torch.from_numpy(z_test[:, np.newaxis]).float())
    #     PE_total_test = PE_normalizer.encode(torch.from_numpy(PE_total_test[:, np.newaxis, np.newaxis]).float())

    # x_train = (torch.cat([x_train, y_train, z_train], dim=1), PE_total_train)
    # x_test = (torch.cat([x_test, y_test, z_test], dim=1), PE_total_test)
    # y_train = torch.from_numpy(Evis_train[:, np.newaxis]).float()
    # y_test = torch.from_numpy(Evis_test[:, np.newaxis]).float()
    
    
    # simulate(model, x_normalizer, y_normalizer, z_normalizer, PE_normalizer)
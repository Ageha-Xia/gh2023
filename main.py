import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from net import MLP
from utils import LpLoss
from timeit import default_timer

n_train = 1000
n_test = 200

class DeepONet(nn.Module):
    def __init__(self, branch, trunk_layers):
        super(DeepONet, self).__init__()
        if isinstance(branch, list):
            self.branch_net = MLP(branch, type='relu', last_activation=False)
        else:
            structure_dict = {'CNN':CNN()}
            self.branch_net = structure_dict[branch]
        self.trunk_net = MLP(trunk_layers, type='relu', last_activation=True)
        self.bias = nn.Parameter(torch.zeros(1))  

    def forward(self, branch, trunk):
        branch_output = self.branch_net(branch)
        trunk_output = self.trunk_net(trunk)

        output = torch.einsum("bi,bji->bj", branch_output, trunk_output) + self.bias
        return output
    
    def run(self, x_train, y_train, x_test, y_test, batch_size, device, criterion='mse', iterations=50, model=None, mode='train'):
        if criterion == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = LpLoss(size_average=False)
            
        # x_train[0] for branch, x_train[1] for trunk
        data_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train[0].to(device), x_train[1].to(device), y_train.to(device)), batch_size=batch_size, shuffle=False)
        data_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test[0].to(device), x_test[1].to(device), y_test.to(device)), batch_size=batch_size, shuffle=False)

        if model:
            self.model = model.to(device)
        else:
            self.model = self.to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iterations / 10, gamma=0.5)
        myloss = LpLoss(size_average=False)

        if mode == 'train':
            for it in range(iterations):
                # self.model.train()
                t1 = default_timer()
                train_l2 = 0
                train_mse = 0
                for branch, trunk, y in data_train:
                    optimizer.zero_grad()
                    out = self.model(branch, trunk)
                    
                    mse = criterion(out.view(batch_size, -1), y.view(batch_size, -1))
                    mse.backward()
                    
                    # out = y_normalizer.decode(out)
                    # y = y_normalizer.decode(y)
                    loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
                    # loss.backward()
            
                    optimizer.step()
                    train_mse += mse.item()
                    train_l2 += loss.item()
            
                scheduler.step()
            
                # self.model.eval()
                test_l2 = 0.0
                with torch.no_grad():
                    for branch_x, trunk_x, y in data_test:
                        out = self.model(branch_x, trunk_x)
                        # out = y_normalizer.decode(out)
                        test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
            
                train_mse /= n_train
                train_l2/= n_train
                test_l2 /= n_test
            
                t2 = default_timer()
                # print(ep, t2-t1, train_l2, test_l2)
                print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" 
                        % ( it, t2-t1, train_mse, train_l2, test_l2) )
            return self.model
        else:
            test_l2 = 0.0
            with torch.no_grad():
                for branch_x, trunk_x, y in data_test:
                    # branch_x = branch_x.unsqueeze(1).repeat(1, trunk_x.shape[1], 1)
                    out = self.model(branch_x, trunk_x).reshape(batch_size, self.s, self.s)
                    # out = y_normalizer.decode(out)
                    test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
            test_l2 /= n_test

            print("Test l2: %.4f" % (test_l2))

with open('data_train.pkl', 'rb') as f:
    x_train, y_train, z_train, PE_total_train, Ek_train, Evis_train = pickle.load(f)
    
with open('data_test.pkl', 'rb') as f:
    x_test, y_test, z_test, PE_total_test, Ek_test, Evis_test = pickle.load(f)

x_train = (torch.from_numpy(np.column_stack((x_train, y_train, z_train))).float(), torch.from_numpy(PE_total_train[:, np.newaxis, np.newaxis]).float())
x_test = (torch.from_numpy(np.column_stack((x_test, y_test, z_test))).float(), torch.from_numpy(PE_total_test[:, np.newaxis, np.newaxis]).float())
y_train = torch.from_numpy(Ek_train[:, np.newaxis])
y_test = torch.from_numpy(Ek_test[:, np.newaxis])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # CNN architecture based on the given TensorFlow code
        self.model = nn.Sequential(
            nn.Unflatten(1, (1, 29, 29)),  # Reshape to (29, 29, 1). Equivalent to Reshape((29, 29, 1)) in TensorFlow
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*128, 128),  # 8x8 is the spatial size after the two convolutions with stride 2 on a 29x29 input
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.model(x)
    
model = DeepONet(branch=[3, 128, 128, 128],
                 trunk_layers=[1, 128, 128, 128])
model.run(x_train, y_train, x_test, y_test, batch_size=20, device='cpu', mode='train', criterion='l2')


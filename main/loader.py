import numpy as np
import h5py
import pickle
from tqdm import tqdm

dataset_count = 1        # 常数尽可能不要硬编码出现在代码里
dataset_prefix = '../data/'  # data文件的位置，我将训练集、测试集放在了data文件夹里

# 第一件事是要确定每个dataset中有多少个事件。这样的好处在于不用硬编码，
# 可以灵活应对dataset中的各种可能的意外情况。
event_count = np.zeros(dataset_count, dtype=int)

# 分数据集读取
for data_id in range(dataset_count):
    with h5py.File(f'{dataset_prefix}{519+data_id}.h5', 'r') as data_file: # python语法f string
        event_count[data_id] = data_file['ParticleTruth'].shape[0]

event_total = event_count.sum() #训练集一共有190000个事件

print(f'训练集一共有{event_total}个事件')

# 初始化固定大小的数组
Ek_train = np.zeros(event_total)
Evis_train = np.zeros(event_total)
PE_total_train = np.zeros(event_total, dtype=int)
x_train = np.zeros(event_total)
y_train = np.zeros(event_total)
z_train = np.zeros(event_total)

# 每一次读数据都需要知道读出来的这一部分在全体数组中的位置，这个数组可以方便索引。
event_index = np.insert(np.cumsum(event_count), 0, 0)

for data_id in tqdm(range(dataset_count)): # tqdm把iterator包起来，就可以实现进度条
    with h5py.File(f'{dataset_prefix}{519+data_id}.h5', 'r') as data_file:
        Ek_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['Ek'][...]
        Evis_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['Evis'][...]
        x_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['x'][...]
        y_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['y'][...]
        z_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['z'][...]
        
        EventIDs = data_file['PETruth']['EventID'][...]
        # 使用np.unique函数避免for循环，"_"是placeholder表示不关心这个变量
        _, PE_total = np.unique(EventIDs, return_counts=True) 
        PE_total_train[event_index[data_id]:event_index[data_id+1]] = PE_total
        
data =  (x_train, y_train, z_train, PE_total_train, Ek_train, Evis_train)
with open('../data/data_test.pkl', 'wb') as file:
    pickle.dump(data, file)
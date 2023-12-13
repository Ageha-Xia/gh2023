import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import Process, Queue
from pyinstrument import Profiler

dataset_count = 19        
dataset_prefix = '../data/'
event_count = np.zeros(dataset_count, dtype=int)
for data_id in range(dataset_count):
    with h5py.File(f'{dataset_prefix}{16930+data_id}.h5', 'r') as data_file: 
        event_count[data_id] = data_file['ParticleTruth'].shape[0]

event_total = event_count.sum()

with h5py.File(f'{dataset_prefix}geo.h5', 'r') as geo_file:
    ChannelID_ = geo_file['Geometry']['ChannelID']
    theta_ = geo_file['Geometry']['theta']
    phi_ = geo_file['Geometry']['phi'] - 180 # let phi in [-180, 180]
    geo_dict = {key: (val1, val2) for key, val1, val2 in zip(ChannelID_, theta_, phi_)}

def process_data(data_id):
    # vecotrize the lookup function of geo_dict
    event_index = np.insert(np.cumsum(event_count), 0, 0)
    vectorized_lookup = np.vectorize(geo_dict.get)
    
    with h5py.File(f'{dataset_prefix}{16930+data_id}.h5', 'r') as data_file:
        
        Ek_train = data_file['ParticleTruth']['Ek'][...]
        Evis_train = data_file['ParticleTruth']['Evis'][...]
        X = data_file['ParticleTruth']['x'][...]
        Y = data_file['ParticleTruth']['y'][...]
        Z = data_file['ParticleTruth']['z'][...]
        
        EventIDs_ = data_file['PETruth']['EventID'][...]
        _, PE_total = np.unique(EventIDs_, return_counts=True) 
        PE_total_train = PE_total
        
        ChannelID_ = data_file['PETruth']['ChannelID'][...]
        PETime_ = data_file['PETruth']['PETime'][...]
        
        ArrivalTime = []
        ArrivalCount = []
        
        for event_id in range(event_count[data_id]):
            # profiler = Profiler()
            # profiler.start()
            
            indices = np.where(EventIDs_ == event_id)
            geo_info = vectorized_lookup(ChannelID_[indices])
            time_info = PETime_[indices]
            event_info = np.column_stack((geo_info[0], geo_info[1] * np.sin(geo_info[0] / 180 * np.pi), time_info))
            
            # convert data to DataFrame
            df = pd.DataFrame(event_info, columns=['Latitude', 'Longitude', 'Value'])

            # calculate mean and count
            grouped = df.groupby(['Latitude', 'Longitude']).agg(['mean', 'count'])
            grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

            ArrivalTime.append(grouped[['Value_mean']].reset_index().to_numpy())
            ArrivalCount.append(grouped[['Value_count']].reset_index().to_numpy())
            

    pos = np.column_stack((X, Y, Z))
    dataset = (Ek_train, Evis_train, pos, PE_total_train, ArrivalTime, ArrivalCount)
    return dataset
    
def get_data():
    # Read the geometry of detector
    
    args = range(dataset_count)    
    with ProcessPoolExecutor() as executor:
        datasets = list(executor.map(process_data, args))
        datasets = [
            np.concatenate([datasets[i][0] for i in range(dataset_count)]),
            np.concatenate([datasets[i][1] for i in range(dataset_count)]),
            np.concatenate([datasets[i][2] for i in range(dataset_count)]),
            np.concatenate([datasets[i][3] for i in range(dataset_count)]),
            [ArrivalTime for i in range(dataset_count) for ArrivalTime in datasets[i][4]],
            [ArrivalCount for i in range(dataset_count) for ArrivalCount in datasets[i][5]],
        ]
        
    with open(f'{dataset_prefix}scatters.pkl', 'wb') as f:
        pickle.dump(datasets, f)

def green_func_torch(scatter, x_range=(-180, 180), y_range=(0, 180), grid_size=(128, 128), device='cuda'):
    x = torch.linspace(x_range[0], x_range[1], grid_size[0], device=device)
    y = torch.linspace(y_range[0], y_range[1], grid_size[1], device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    scatter_tensor = torch.from_numpy(scatter).float().to(device)

    x_diff = xx.unsqueeze(-1) - scatter_tensor[:, 1].unsqueeze(0).unsqueeze(0)
    y_diff = yy.unsqueeze(-1) - scatter_tensor[:, 0].unsqueeze(0).unsqueeze(0)
    r = torch.sqrt(x_diff**2 + y_diff**2)
    v = scatter_tensor[:, 2] * torch.exp(-r**2)

    zz = v.sum(dim=-1)
    return zz.cpu().numpy()


def process_data_on_gpu(gpu_id, data_queue, result_queue, chunk_size):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    while True:
        data_block = data_queue.get()
        if data_block is None:
            break  # 结束信号
        
        ArrivalTime = data_block[0]
        ArrivalCount = data_block[1]
        
        # 对于每个 GPU 接收到的数据块进行处理
        processed_data = []
        for i in range(chunk_size):
            # 处理每个数据块
            ArrivalTimeImage = green_func_torch(ArrivalTime[i], device=device)
            ArrivalCountImage = green_func_torch(ArrivalCount[i], device=device)
            processed_data.append((ArrivalTimeImage, ArrivalCountImage))
        
        # 将处理后的数据块的结果放入结果队列
        result_queue.put(processed_data)

def generate_image(gpu_count, chunk_size=10000):
    with open(f'{dataset_prefix}scatters.pkl', 'rb') as f:
        Ek_train, Evis_train, pos, PE_total_train, ArrivalTime, ArrivalCount = pickle.load(f)

    data_queues = [Queue() for _ in range(gpu_count)]
    result_queue = Queue()
    processes = []

    # 启动处理进程
    for gpu_id in range(gpu_count):
        p = Process(target=process_data_on_gpu, args=(gpu_id, data_queues[gpu_id], result_queue, chunk_size))
        p.start()
        processes.append(p)

    # 分配数据给各 GPU
    for i in range(0, len(ArrivalTime), chunk_size):
        gpu_id = i // chunk_size % gpu_count
        data_queues[gpu_id].put((ArrivalTime[i:i + chunk_size], ArrivalCount[i:i + chunk_size]))

    # 发送结束信号
    for q in data_queues:
        q.put(None)

    # 收集结果
    EventImage = []
    for _ in range(event_total // chunk_size):
        processed_data_block = result_queue.get()
        # 扁平化每个数据块的结果，并将它们添加到 EventImage
        EventImage.extend(processed_data_block)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 保存结果
    with open(f'{dataset_prefix}eventimage.pkl', 'wb') as f:
        pickle.dump(EventImage, f)

        
if __name__ == '__main__':
    if not os.path.exists(f'{dataset_prefix}scatters.pkl'):
        get_data()
    gpu_count = torch.cuda.device_count()  # 检测可用的 GPU 数量
    generate_image(gpu_count)
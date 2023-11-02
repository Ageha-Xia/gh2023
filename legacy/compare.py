import numpy as np
import h5py
from matplotlib import pyplot as plt              # matplotlib是常用于画图的包
from tqdm import tqdm                             # tqdm可以显示进度条
from sklearn.linear_model import LinearRegression # 这里利用了sklearn的线性拟合
dataset_count = 18        # 常数尽可能不要硬编码出现在代码里
dataset_prefix = 'data/'  # data文件的位置，我将训练集、测试集放在了data文件夹里

if __name__ == '__main__':
    
    event_total = 10000
# 初始化固定大小的数组
    Ek_train = np.zeros(event_total)
    Event_pos_x = np.zeros(event_total)
    Event_pos_y = np.zeros(event_total)
    Event_pos_z = np.zeros(event_total)
    Event_pos_r = np.zeros(event_total)
    Event_theta = np.zeros(event_total)
    Event_phi = np.zeros(event_total)
    #detector_theta= np.zeros(event_total)
    #detector_phi = np.zeros(event_total)
    
    my_Ek_train = np.zeros(event_total)
    
# 每一次读数据都需要知道读出来的这一部分在全体数组中的位置，这个数组可以方便索引。
# 这个数组存的是0，10000，20000...,190000
    
    with h5py.File(f'{dataset_prefix}{519}.h5', 'r') as data_file:
        # 这句话表示为Ek_train从下标event_index[data_id]到event_index[data_id+1]赋值
        Ek_train = data_file['ParticleTruth']['Ek'][...]
            
        Event_pos_x = data_file['ParticleTruth']['x'][...]
        Event_pos_y = data_file['ParticleTruth']['y'][...]
        Event_pos_z = data_file['ParticleTruth']['z'][...]
        
        
        EventIDs = data_file['PETruth']['EventID'][...]
        # 使用np.unique函数避免for循环，"_"是placeholder表示不关心这个变量
        # 这里是在数相同的EventID发生了多少次
        _, PE_total = np.unique(EventIDs, return_counts=True) 
        E_0 = 0.511 # 电子的静能
        Event_pos_r = (Event_pos_x * Event_pos_x + Event_pos_y * Event_pos_y + Event_pos_z * Event_pos_z)**0.5
        #Event_phi = math.atan(Event_pos_y / Event_pos_x) / math.pi * 180
        #Event_theta = math.asin(Event_pos_z / Event_pos_r) / math.pi * 180
    
    with h5py.File('519_my.h5', 'r') as data_file:
        # 这句话表示为Ek_train从下标event_index[data_id]到event_index[data_id+1]赋值
        my_Ek_train = data_file['Answer']['Ek'][...]
    
        
    dif = (my_Ek_train - Ek_train)
    #print(np.max(Event_pos_r))
    #print(np.max(Event_pos_r) - np.min(Event_pos_r.min))
    lower_bound = np.min(Event_pos_r)
    
    interval = np.max(Event_pos_r) - np.min(Event_pos_r)
    Scaled_Event_pos_r = np.zeros(event_total)    
    Scaled_Event_pos_r = (Event_pos_r - lower_bound) / (interval)
    
    plt.rcParams['figure.figsize'] = (15.0, 5.0) # 设置一下图片的大小（全局设置），这个不是唯一的方法
    
    #条件筛选
    #Selected_pos = np.where((Scaled_Event_pos_r > 0.98) | (Scaled_Event_pos_r < 0.02))
    #Selected_PE_total_train = PE_total_train[Selected_pos]
    
    plt.scatter(Event_pos_r, dif, s=4)
    plt.show()                                   # 这样正常来说会弹出一个窗口显示画的图，如果没有请查询相关教程
    print(sum(dif * dif))
    # 先创建一个LinearRegression的Instance，注意这里不拟合截距。然后调用fit方法
    # 注意这个fit函数要求自变量是二维数组，因为一般情况下自变量可以有很多个。这里自变量只有1个，因此使用reshape函数强行变成二维。
import numpy as np
import h5py
from matplotlib import pyplot as plt              # matplotlib是常用于画图的包
from tqdm import tqdm                             # tqdm可以显示进度条
from sklearn.linear_model import LinearRegression # 这里利用了sklearn的线性拟合

dataset_count = 19        # 常数尽可能不要硬编码出现在代码里
dataset_prefix = '../data/'  # data文件的位置，我将训练集、测试集放在了data文件夹里

# 第一件事是要确定每个dataset中有多少个事件。这样的好处在于不用硬编码，
# 可以灵活应对dataset中的各种可能的意外情况。
event_count = np.zeros(dataset_count, dtype=int)

# 分数据集读取
for data_id in range(dataset_count):
    with h5py.File(f'{dataset_prefix}{501+data_id}.h5', 'r') as data_file: # python语法f string
        event_count[data_id] = data_file['ParticleTruth'].shape[0]

event_total = event_count.sum() #训练集一共有190000个事件

print(f'训练集一共有{event_total}个事件')

# 初始化固定大小的数组
Ek_train = np.zeros(event_total)
Evis_train = np.zeros(event_total)
PE_total_train = np.zeros(event_total, dtype=int)

# 每一次读数据都需要知道读出来的这一部分在全体数组中的位置，这个数组可以方便索引。
event_index = np.insert(np.cumsum(event_count), 0, 0)

for data_id in tqdm(range(dataset_count)): # tqdm把iterator包起来，就可以实现进度条
    with h5py.File(f'{dataset_prefix}{501+data_id}.h5', 'r') as data_file:
        Ek_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['Ek'][...]
        Evis_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['Evis'][...]
        
        EventIDs = data_file['PETruth']['EventID'][...]
        # 使用np.unique函数避免for循环，"_"是placeholder表示不关心这个变量
        _, PE_total = np.unique(EventIDs, return_counts=True) 
        PE_total_train[event_index[data_id]:event_index[data_id+1]] = PE_total
        
E_0 = 0.511 # 电子的静能

plt.rcParams['figure.figsize'] = (15.0, 5.0) # 设置一下图片的大小（全局设置），这个不是唯一的方法

fig, ax = plt.subplots(1,3)                  # 分成三个图，subplots前一个参数是行数，后一个是列数。
ax[0].scatter(PE_total_train, Ek_train+E_0*2)# ax[0]表示第一个图，scatter是散点图，后面跟x坐标和y坐标。
ax[0].set_xlabel('total PE')                 # 注意中文字体可能会有问题，如果需要的话请查询相关教程
ax[0].set_ylabel('Ek+2E0/MeV')               # 其实是支持LaTeX的，我只是懒得搞
ax[1].scatter(PE_total_train, Evis_train)
ax[1].set_xlabel('total PE')
ax[1].set_ylabel('Evis/MeV')
ax[2].scatter(Evis_train, Ek_train+E_0*2)
ax[2].set_xlabel('Evis/MeV')
ax[2].set_ylabel('Ek+2E0/MeV')

plt.show()                                   # 这样正常来说会弹出一个窗口显示画的图，如果没有请查询相关教程

# 先创建一个LinearRegression的Instance，注意这里不拟合截距。然后调用fit方法
# 注意这个fit函数要求自变量是二维数组，因为一般情况下自变量可以有很多个。这里自变量只有1个，因此使用reshape函数强行变成二维。
model_Ek = LinearRegression(fit_intercept=False).fit(PE_total_train.reshape(-1,1), Ek_train+E_0*2)
model_Evis = LinearRegression(fit_intercept=False).fit(PE_total_train.reshape(-1,1), Evis_train)

# 不再赘述了，如果对于h5文件读写不熟悉，请观看宣讲会playground讲解部分的录像。
with h5py.File(f'{dataset_prefix}problem.h5', 'r') as problem_file:
    EventIDs = problem_file['PETruth']['EventID']
    EventID_problem, PE_total_problem = np.unique(EventIDs, return_counts=True)

Ek_problem = model_Ek.predict(PE_total_problem.reshape(-1,1))-E_0*2
Evis_problem = model_Evis.predict(PE_total_problem.reshape(-1,1))

ans_dtype = np.dtype([
    ('EventID', '<i4'),
    ('Ek', '<f4'),
    ('Evis', '<f4')
])
ans_data = np.zeros(Ek_problem.shape, dtype=ans_dtype)
ans_data['EventID'], ans_data['Ek'], ans_data['Evis'] = EventID_problem, Ek_problem, Evis_problem

with h5py.File(f'ans.h5', 'w') as answer_file:
    answer_file.create_dataset('Answer', data=ans_data)
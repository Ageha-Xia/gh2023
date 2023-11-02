import numpy as np
import h5py
from matplotlib import pyplot as plt              # matplotlib是常用于画图的包
from tqdm import tqdm                             # tqdm可以显示进度条
from sklearn.linear_model import LinearRegression # 这里利用了sklearn的线性拟合
from sklearn.preprocessing import PolynomialFeatures #利用sklearn的二次拟合
import bisect
dataset_count = 19        # 常数尽可能不要硬编码出现在代码里
dataset_prefix = '../data/'  # data文件的位置，我将训练集、测试集放在了data文件夹里
def getCoef(index):
    
    return index

if __name__ == '__main__':
    

# 第一件事是要确定每个dataset中有多少个事件。这样的好处在于不用硬编码，
# 可以灵活应对dataset中的各种可能的意外情况。
    event_count = np.zeros(dataset_count, dtype=int)

# 分数据集读取
    for data_id in range(dataset_count):
        with h5py.File(f'{dataset_prefix}{501+data_id}.h5', 'r') as data_file: # python语法f string
            event_count[data_id] = data_file['ParticleTruth'].shape[0] #shape[0]返回有多少个元组

    event_total = event_count.sum()# 训练集一共有190000个事件

#    for event_num in event_count:
#        print(event_num, end=' ')
    print(f'训练集一共有{event_total}个事件')
# 初始化固定大小的数组
    Ek_train = np.zeros(event_total)
    Evis_train = np.zeros(event_total)
    PE_total_train = np.zeros(event_total, dtype=int)

    Event_pos_x = np.zeros(event_total)
    Event_pos_y = np.zeros(event_total)
    Event_pos_z = np.zeros(event_total)
    Event_pos_r = np.zeros(event_total)
    
# 每一次读数据都需要知道读出来的这一部分在全体数组中的位置，这个数组可以方便索引。
# 这个数组存的是0，10000，20000...,190000
    event_index = np.insert(np.cumsum(event_count), 0, 0)

    for data_id in tqdm(range(dataset_count)): # tqdm把iterator包起来，就可以实现进度条
        with h5py.File(f'{dataset_prefix}{501+data_id}.h5', 'r') as data_file:
            # 这句话表示为Ek_train从下标event_index[data_id]到event_index[data_id+1]赋值
            Ek_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['Ek'][...]
            Evis_train[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['Evis'][...]
            
            Event_pos_x[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['x'][...]
            Event_pos_y[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['y'][...]
            Event_pos_z[event_index[data_id]:event_index[data_id+1]] = data_file['ParticleTruth']['z'][...]

        
            EventIDs = data_file['PETruth']['EventID'][...]
        # 使用np.unique函数避免for循环，"_"是placeholder表示不关心这个变量
        # 这里是在数相同的EventID发生了多少次
            _, PE_total = np.unique(EventIDs, return_counts=True) 
            PE_total_train[event_index[data_id]:event_index[data_id+1]] = PE_total
            E_0 = 0.511 # 电子的静能
            Event_pos_r[event_index[data_id]:event_index[data_id+1]] = (Event_pos_x[event_index[data_id]:event_index[data_id+1]] ** 2 
                                                                      + Event_pos_y[event_index[data_id]:event_index[data_id+1]] ** 2
                                                                      + Event_pos_z[event_index[data_id]:event_index[data_id+1]] ** 2) ** 0.5
    
    #print(np.max(Event_pos_r))
    #print(np.max(Event_pos_r) - np.min(Event_pos_r.min))
    '''
    lower_bound = np.min(Event_pos_r)
    
    interval = np.max(Event_pos_r) - np.min(Event_pos_r)
    Scaled_Event_pos_r = np.zeros(event_total)    
    Scaled_Event_pos_r = (Event_pos_r - lower_bound) / (interval)
    
    plt.rcParams['figure.figsize'] = (15.0, 5.0) # 设置一下图片的大小（全局设置），这个不是唯一的方法
    
    #条件筛选
    Selected_pos = np.where((Scaled_Event_pos_r > 0.98) | (Scaled_Event_pos_r < 0.02))
    #Selected_PE_total_train = PE_total_train[Selected_pos]
    
    
    fig, ax = plt.subplots(1,3)                  # 分成三个图，subplots前一个参数是行数，后一个是列数。
    ax[0].scatter(PE_total_train[Selected_pos], Ek_train[Selected_pos]+E_0*2, c = Scaled_Event_pos_r[Selected_pos])# ax[0]表示第一个图，scatter是散点图，后面跟x坐标和y坐标。
    ax[0].set_xlabel('total PE')                 # 注意中文字体可能会有问题，如果需要的话请查询相关教程
    ax[0].set_ylabel('Ek+2E0/MeV')               # 其实是支持LaTeX的，我只是懒得搞
    ax[1].scatter(PE_total_train, Evis_train, c = Scaled_Event_pos_r)
    ax[1].set_xlabel('total PE')
    ax[1].set_ylabel('Evis/MeV')
    ax[2].scatter(Evis_train, Ek_train+E_0*2, c = Scaled_Event_pos_r)
    ax[2].set_xlabel('Evis/MeV')
    ax[2].set_ylabel('Ek+2E0/MeV')

    #plt.show()                                   # 这样正常来说会弹出一个窗口显示画的图，如果没有请查询相关教程
    '''
    # 先创建一个LinearRegression的Instance，注意这里不拟合截距。然后调用fit方法
    # 注意这个fit函数要求自变量是二维数组，因为一般情况下自变量可以有很多个。这里自变量只有1个，因此使用reshape函数强行变成二维。
     
    data = np.vstack((Event_pos_r, PE_total_train, Ek_train))
    data = data.T[np.lexsort(data[::-1, :])].T 
    
    Event_pos_r = data[0, :]
    PE_total_train = data[1, :]
    Ek_train = data[2, :]
    
    N = 500
    #每N个Event_pos_r取一个值，作为与问题比较的下标
    Event_pos_r_label = np.zeros(int(event_total / N))
    quadratic_coef = np.zeros((int(event_total/N), 3))
    r2 = np.zeros(int(event_total / N))
    for i in range(int(event_total / N)):
        Event_pos_r_label[i] = Event_pos_r[i * N]
        #线性拟合
        model_Ek = LinearRegression(fit_intercept=False).fit(PE_total_train[i * N : (i + 1) * N].reshape(-1,1), Ek_train[i * N : (i + 1) * N])
        quadratic_featurizer = PolynomialFeatures(degree=2)
        X_train_quadratic = quadratic_featurizer.fit_transform(PE_total_train[i * N : (i + 1) * N].reshape(-1, 1))
        regressor_quadratic = LinearRegression(fit_intercept=False)
        regressor_quadratic.fit(X_train_quadratic, Ek_train[i * N : (i + 1) * N])
        #print(regressor_quadratic.coef_)
        #print(model_Ek.coef_)
        #xx = np.linspace(0,100,5)
        #print(regressor_quadratic.predict(xx.reshape(-1,1)))
        #print(model_Ek.predict(xx.reshape(-1,1)))
        quadratic_coef[i] = regressor_quadratic.coef_
        r2[i] = regressor_quadratic.score(X_train_quadratic, Ek_train[i * N : (i + 1) * N])
        
    fig, ax = plt.subplots(2,2)                  # 分成三个图，subplots前一个参数是行数，后一个是列数。
    ax[0][0].scatter(Event_pos_r_label, quadratic_coef[ :,0],s=3)# ax[0]表示第一个图，scatter是散点图，后面跟x坐标和y坐标。
    ax[0][1].scatter(Event_pos_r_label, quadratic_coef[ :,1],s=3)
    ax[1][0].scatter(Event_pos_r_label, quadratic_coef[ :,2],s=3)
    ax[1][1].scatter(Event_pos_r_label, r2, s=3)
    plt.show()                                   # 这样正常来说会弹出一个窗口显示画的图，如果没有请查询相关教程
    
    np.savetxt('.\quadratic_coef.txt', quadratic_coef, fmt='%f',delimiter=',')
    
    problem_event_pos_x = np.zeros(10000)
    problem_event_pos_y = np.zeros(10000)
    problem_event_pos_z = np.zeros(10000)
    problem_event_pos_r = np.zeros(10000)
    # 不再赘述了，如果对于h5文件读写不熟悉，请观看宣讲会playground讲解部分的录像。
    with h5py.File(f'{dataset_prefix}problem.h5', 'r') as problem_file:
        EventIDs = problem_file['PETruth']['EventID']
        EventID_problem, PE_total_problem = np.unique(EventIDs, return_counts=True)
        problem_event_pos_x = problem_file['ParticleTruth']['x'][...]
        problem_event_pos_y = problem_file['ParticleTruth']['y'][...]
        problem_event_pos_z = problem_file['ParticleTruth']['z'][...]
        problem_event_pos_r = (problem_event_pos_x * problem_event_pos_x + problem_event_pos_y * problem_event_pos_y + problem_event_pos_z * problem_event_pos_z)**0.5
       
    #Aproblem_Scaled_Event_pos_r = (problem_event_pos_r - lower_bound) / (interval)
    
    #index = np.floor(problem_Scaled_Event_pos_r * N)
    #index.astype(np.int32)

    Ek_problem = np.zeros(10000)
    Evis_problem = np.zeros(10000)
    
    for i in range(10000):
        idx = bisect.bisect_left(Event_pos_r_label, problem_event_pos_r[i]) - 1
        Ek_problem[i] = quadratic_coef[idx][2] * (PE_total_problem[i]**2) + quadratic_coef[idx][1] * (PE_total_problem[i]) +quadratic_coef[idx][0]
        Evis_problem[i] = Ek_problem[i] + 2 * E_0

    ans_dtype = np.dtype([
        ('EventID', '<i4'),
        ('Ek', '<f4'),
        ('Evis', '<f4')
    ])
    ans_data = np.zeros(Ek_problem.shape, dtype=ans_dtype)
    ans_data['EventID'], ans_data['Ek'], ans_data['Evis'] = EventID_problem, Ek_problem, Evis_problem

    with h5py.File(f'ans.h5', 'w') as answer_file:
        answer_file.create_dataset('Answer', data=ans_data)
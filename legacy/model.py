import h5py
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm


def model(interval):

    filepath = 'E:\\ghost_hunter\\'
    total_num = 19
    Ek = np.zeros(10000 * total_num)
    Evis = np.zeros(10000 * total_num)
    PE = np.zeros(10000 * total_num)
    r = np.zeros(10000 * total_num)

    for _id in range(total_num):
        with h5py.File(f'{filepath}{501 + _id}.h5', 'r') as f:
            _, PE[_id * 10000:(_id + 1) * 10000] = np.unique(f['PETruth']['EventID'], return_counts=True)
            #r[_id * 10000:(_id + 1) * 10000] = abs(f['ParticleTruth']['x'][...])+abs(f['ParticleTruth']['y'][...])+abs(f['ParticleTruth']['z'][...])
            r[_id * 10000:(_id + 1) * 10000] = (f['ParticleTruth']['x'][...] ** 2 + f['ParticleTruth']['y'][...] ** 2 + f['ParticleTruth']['z'][...] ** 2) ** 0.5
            Ek[_id * 10000:(_id + 1) * 10000] = f['ParticleTruth']['Ek'][...]
            Evis[_id * 10000:(_id + 1) * 10000] = f['ParticleTruth']['Evis'][...]


    r_color = 100 / (np.max(r) - np.min(r)) * (r - np.min(r))
    #np.savetxt('E:\\ghost_hunter\\param\\r.txt', [np.max(r), np.min(r)], fmt='%f', delimiter=',')
    data = np.vstack((r_color, PE, Ek))
    data = data.T[np.lexsort(data[::-1, :])].T

    t_interval = interval
    param_size = 10000 * total_num // t_interval
    b0 = np.zeros(param_size)
    b1 = np.zeros(param_size)
    r2 = np.zeros(param_size)
    t_cut = np.zeros(param_size)
    for t in range(param_size):
        fit = sm.OLS(data[2, t * t_interval:(t+1) * t_interval], sm.add_constant(data[1, t * t_interval:(t+1) * t_interval])).fit()
        b0[t] = fit.params[0]
        b1[t] = fit.params[1]
        r2[t] = fit.rsquared
        t_cut[t] = data[0, t * t_interval]
    #pic = plt.figure(figsize=(15, 5))
    #print(b1)
    #print(b0)
    np.savetxt('E:\\ghost_hunter\\param\\b1.txt', b1, fmt='%f', delimiter=',')
    np.savetxt('E:\\ghost_hunter\\param\\b0.txt', b0, fmt='%f', delimiter=',')
    np.savetxt('E:\\ghost_hunter\\param\\t_cut.txt', t_cut, fmt='%f', delimiter=',')
    """
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(t_cut, b1, s=5)
    ax[1].scatter(t_cut, b0, s=5)
    ax[2].scatter(t_cut, r2, s=5)
    #plt.scatter(data[1, :1000], data[2, :1000], s=5, c=data[0, :1000])
    #plt.plot([0, 12], [b0, b0+12*b1], 'r', linewidth=1.5)
    #plt.scatter(np.linspace(t_cyt, b1, s=5)
    plt.show()
    """
    print('model_finish')



"""
    for _id in range(total_num):
        with h5py.File(f'{filepath}{501 + _id}.h5', 'r') as f:
            _, PE[_id * 10000:(_id + 1) * 10000] = np.unique(f['PETruth']['EventID'], return_counts=True)
            #r[_id * 10000:(_id + 1) * 10000] = abs(f['ParticleTruth']['x'][...])+abs(f['ParticleTruth']['y'][...])+abs(f['ParticleTruth']['z'][...])
            r[_id * 10000:(_id + 1) * 10000] = (f['ParticleTruth']['x'][...] ** 2 + f['ParticleTruth']['y'][...] ** 2 + f['ParticleTruth']['z'][...] ** 2) ** 0.5
            Ek[_id * 10000:(_id + 1) * 10000] = f['ParticleTruth']['Ek'][...]
            Evis[_id * 10000:(_id + 1) * 10000] = f['ParticleTruth']['Evis'][...]


    r_color = 100 / (np.max(r) - np.min(r)) * (r - np.min(r))
    data = np.vstack((r_color, PE, Evis))
    data = data.T[np.lexsort(data[::-1, :])].T

    t_num = 25
    b0 = np.zeros(t_num)
    b1 = np.zeros(t_num)
    r2 = np.zeros(t_num)
    for t in range(t_num):
        data_t = data[:, (data[0] >= t*100/t_num) & (data[0] < (t+1)*100/t_num)]
        print(data_t.shape)
        fit = sm.OLS(data_t[2], sm.add_constant(data_t[1])).fit()
        b0[t] = fit.params[0]
        b1[t] = fit.params[1]
        r2[t] = fit.rsquared
    #pic = plt.figure(figsize=(15, 5))
    print(b1)
    print(b0)
    np.savetxt('E:\\ghost_hunter\\param\\b1.txt', b1, fmt='%f', delimiter=',')
    np.savetxt('E:\\ghost_hunter\\param\\b0.txt', b0, fmt='%f', delimiter=',')
    np.savetxt('E:\\ghost_hunter\\param\\t_cut.txt', t_cut, fmt='%f', delimiter=',')
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(np.linspace(50/t_num, 100-50/t_num, t_num), b1, s=8)
    ax[1].scatter(np.linspace(50 / t_num, 100 - 50 / t_num, t_num), b0, s=8)
    ax[2].scatter(np.linspace(50 / t_num, 100 - 50 / t_num, t_num), r2, s=8)
    #plt.scatter(data[1, :1000], data[2, :1000], s=5, c=data[0, :1000])
    #plt.plot([0, 12], [b0, b0+12*b1], 'r', linewidth=1.5)
    plt.scatter(np.linspace(50/t_num, 100-50/t_num, t_num), b1, s=5)
    plt.show()
    print('finish')

"""
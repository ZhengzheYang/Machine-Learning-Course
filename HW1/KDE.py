from data_loader import DataLoader
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

#load data
data = DataLoader()
data.load()

trn_0 = data.trn_0
trn_1 = data.trn_1
tst_0 = data.tst_0
tst_1 = data.tst_1
trn_all = data.trn_all
tst_all = data.tst_all

values_0 = np.vstack([trn_0[0], trn_0[1]])
values_1 = np.vstack([trn_1[0], trn_1[1]])

count = 0
for bw in (0.1, 1, 10):
    fig = plt.figure()
    kde_1 = gaussian_kde(values_0, bw_method=bw)
    kde_2 = gaussian_kde(values_1, bw_method=bw)
    for i, data in enumerate(tst_all[0:2].T):
        # print(data[0:2])
        if kde_1(data) > kde_2(data):
            classes = 0
        else:
            classes = 1
        color = 'blue'
        marker = 'o'
        if classes != tst_all[2][i]:
            count += 1
            color = 'red'
            marker = 'x'
        plt.plot(data[0], data[1], color=color, marker=marker)
        # print(data)
    print("The classification error rate for bandwidth = %s is %f" %(bw, float(count)/tst_all.shape[1]))
plt.show()



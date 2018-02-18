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


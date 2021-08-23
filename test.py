import scipy.io as scio
name_mat = scio.loadmat("E:\CUZ2021\\imgs.mat")
imgs=name_mat['imgs']
label_mat = scio.loadmat("D:\ImageDatabase\\tid2013\\OBSCsn.mat")
lbs = label_mat['OBSCsn']
a=10
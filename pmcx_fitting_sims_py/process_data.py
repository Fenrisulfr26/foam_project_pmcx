# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:00:08 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from cv2 import resize
import cv2

#%% the data

st_crop = 100
bins = 700

IRF = np.load(r'../plastock_foam_light\XL_measurement_061224/IRF_gain0.7_timebin15ps_2000bins_061224.npy')[:1000]
IRF = IRF - IRF[:100].mean()
IRF = IRF[st_crop:st_crop+700]
IRF = IRF/IRF.max()

## Downsample the irf to convolve with the results of FEM
data_irf_down = resize(IRF[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)
data_irf_down = np.squeeze(data_irf_down) # get rid of extra dimension
data_irf_down = data_irf_down/data_irf_down.max()

## no absorption data
noabs = np.load(r'../plastock_foam_light\XL_measurement_061224/noabs1_gain0.7_5x5cm_31x31points_pol50deg_expo1.0sec_binWidth15ps_binNum2000_061224.npy')[:,:,:1000]
irf_peak = np.argmax(IRF)-np.round((5e-2/const.c)/15e-12)
irf_peak = int(irf_peak)

noabs = noabs[:,:,st_crop+irf_peak:st_crop+irf_peak+bins]
X,Y = np.meshgrid([0,15,30], [0,15,30])
noabs = noabs[X,Y]
noabs = noabs/noabs.max()

no_abs_down = np.zeros((3, 3, 100+1))
for i in range(3):
    for j in range(3):
        no_abs_down[i,j] = np.squeeze(resize(noabs[i,j,:667], (1, 101), 
                                              interpolation=cv2.INTER_NEAREST_EXACT))
no_abs_down = no_abs_down/no_abs_down.max()

## time
time = np.linspace(0,(noabs.shape[2]-1)*(15e-3), noabs.shape[2])
time_down = resize(time[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)

data_processed = {'noabs_processed': no_abs_down,
                'IRF_processed': data_irf_down,
                'time_processed': time_down}
np.save('noabs_processed.npy', data_processed)

data = np.load('noabs_processed.npy', allow_pickle=1)[()]

noabs = data['noabs_processed']
IRF = data['IRF_processed']
time = data['time_processed']

#%%

plt.plot(time, IRF)
plt.plot(time, noabs.reshape(-1,noabs.shape[2]).T)

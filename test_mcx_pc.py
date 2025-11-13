# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:20:35 2025

@author: zhiguan wang
"""

#%% test different mcx phton number 

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pmcx
from scipy.optimize import curve_fit
from cv2 import resize
import cv2
from scipy.signal import fftconvolve


params= dict(mu_a=0.03730923634486534,         # absorption coefficient (cm^-1)
           mu_s=16.873585007998496,         # reduced scattering coefficient (cm^-1)
           n=1.44,            #refractive index of material
           slab_1=5,        # thickness of slab 1 (cm)
           slab_2=0,        # thickness of slab 2 (cm)
            FOV=9,           # square FOV of the camera (cm) 
            # FOV=2.2,           # square FOV of the camera (cm) 
           res=51,            # Resolution of the camera (make an odd number)
           pad_dim = 64,    # padding resolution outwith FOV (make an even number)
           c=const.c*100,     # speed of light in cm/s
           bins=700,          # Number of timebins (max st_bin=120)
           t_res=15e-12,      # time resolution of the PMT
           pad_time= 100,       #padding in time (only important for very late time traces)
           beam_pos=[0,0],    # center point of the incident beam (cm)
           st_dev=.25,        # beam width (cm)
           pulse_st=0,        # starting bin for the Gaussian pulse 
           st_bin=0,         # starting bin of histogram 
           nd_bin=700,        # end bin of histgram
           lamb_grad = 1 ,    # set to 1 or zero to include derivative of (A(X)-Y) (this is used to troubleshoot regularisers)
           v_offset=0, # offset in vertical position of sources
           h_offset=0 # offset in horizonatal position of sources
           )

#%%
# IRF = np.load(r'../plastock_foam_light\XL_measurement_061224/IRF_gain0.7_timebin15ps_2000bins_061224.npy')[:1000]
IRF = np.load(r'IRF_gain0.7_timebin15ps_2000bins_061224.npy')[:1000]
IRF = IRF - IRF[:100].mean()
IRF = IRF[100:]
IRF = IRF/IRF.max()
IRF = IRF[:params['bins']]

irf_peak = np.argmax(IRF)-np.round(((params['slab_1']+params['slab_2'])*1e-2/const.c)/params['t_res'])
irf_peak = int(irf_peak)

# noabs = np.load(r'../plastock_foam_light\XL_measurement_061224/noabs1_gain0.7_5x5cm_31x31points_pol50deg_expo1.0sec_binWidth15ps_binNum2000_061224.npy')[:,:,:1000]
noabs = np.load(r'noabs1_gain0.7_5x5cm_31x31points_pol50deg_expo1.0sec_binWidth15ps_binNum2000_061224.npy')[:,:,:1000]
noabs = noabs[:,:,100+irf_peak:100+irf_peak+params['bins']]
X,Y = np.meshgrid([0,15,30], [0,15,30])
no_abs = noabs[X,Y]

# downsample the data
time = np.linspace(0,(no_abs.shape[2]-1)*(15e-3), no_abs.shape[2])

## Downsample the irf to convolve with the results of MCX
data_irf_down = resize(IRF[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)
data_irf_down = np.squeeze(data_irf_down) # get rid of extra dimension
time_down = resize(time[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)

no_abs_down = np.zeros((3, 3, 100+1))
for i in range(3):
    for j in range(3):
        no_abs_down[i,j] = np.squeeze(resize(no_abs[i,j,:667], (1, 101), 
                                             interpolation=cv2.INTER_NEAREST_EXACT))

params['IRF'] =  data_irf_down

plt.figure()
plt.plot(data_irf_down)
plt.show()
#%%

thickness = 50 # mm
unitinmm = 1
t_vox = int(thickness/unitinmm)
vol = np.ones([250,250,t_vox+1],dtype='uint8')
vol[:,:,-1] = 0

cfg = {'nphoton': 1e7, 
       'vol': vol, 
       'tstart':0, 
        'tend':10.1e-9, 
        'tstep':0.1e-9,  # 100ps resolution
       'srcpos': [125,125,0], 
       'srcdir':[0,0,1],
       'unitinmm':unitinmm,
       'detpos': [125, 125, t_vox, 4], #radius approx 1mm (4vox) (fibre ~2mm from sample)
       'issrcfrom0':1,
       'issavedet':1,
       'issaveref':1
       }

## make sensitivity profile for sims
# sensitivity = np.load('../plastock_foam_light/XL_measurement_061224/sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy').sum(2)
sensitivity = np.load('sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy').sum(2)
sensitivity = sensitivity/sensitivity.max()
sensitivity = np.fliplr(sensitivity)
sensitivity = resize(sensitivity, (80,80), interpolation=cv2.INTER_LINEAR)
sens = np.zeros((250,250))
sens[125-40:125+40, 125-40:125+40] = sensitivity

plt.figure()
plt.imshow(sens)
plt.show()

#%% 

def simfn(cfg, mu_a, mu_s):
    cfg['prop'] = [[0,0,1,1],           # background
                   [mu_a,mu_s,0,1.44]  # volume 1
                   ]
    
    # return pmcx.mcxlab(cfg)   # pmcx.mcxlab calls pmcx.run, and postprocess res['detp'] and res['traj'] raw data into dict form
    return pmcx.run(cfg)   # pmcx.mcxlab calls pmcx.run, and postprocess res['detp'] and res['traj'] raw data into dict form

def fitfn(cfg, mu_a, mu_s):
    
    sim_3x3 = np.zeros((3,3,101))
    for i, posx in enumerate([100, 125, 150]):
        for j, posy in enumerate([100, 125, 150]):
            cfg['srcpos'] = [posx, posy, 0]
            print(cfg['srcpos'])
            
            res = simfn(cfg, mu_a, mu_s)
            
            # retrieve the results via surface fluence
            sim = (res['dref'][:,:,int(vol.shape[2]-1),:]*sens[:,:,None]).sum((0,1))
            sim_3x3[i, j] = sim
            
            print(f'mua = {mu_a}\nmus = {mu_s}')
    
    IRF = params['IRF']
    
    conv_simp = np.apply_along_axis(lambda m: fftconvolve(m, IRF, mode='full'), axis=2, arr=sim_3x3)[:,:,:len(IRF)]
    conv_simp = conv_simp/conv_simp.max()
    conv_simp = conv_simp[:,:,:101]

    return conv_simp

#%%

cfg1 = cfg
cfg2 = cfg
cfg2['nphoton'] = 1e8

fit_sim1 = fitfn(cfg1,0.01,0.5)
# fit_sim2 = fitfn(cfg2,0.01,1)

plt.figure()
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        plt.plot(fit_sim1[i,j,:], label='fit')
        # plt.plot((fit_sim2)[i,j,:], label='ground truth')
        plt.ylim([0,1.2])
        # plt.title(', '.join(f'{x:.4f}' for x in p))
plt.legend()
plt.show()

#%%
fit_sim1_slip = np.flip(fit_sim1, axis = 1)
plt.figure()
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        plt.plot(no_abs_down[i,j,:]/no_abs_down.max())
        plt.plot(fit_sim1_slip[i,j,:])
        plt.ylim([0,1.2])
        # plt.title(', '.join(f'{x:.4f}' for x in p))
plt.legend()
plt.show()

#%% additional distance by scan
cfg['prop'] = [[0,0,1,1],           # background
               [0.01,1,0,1.44]  # volume 1
               ]
cfg['nphoton'] = 1e8
res = pmcx.run(cfg)

plt.figure()
plt.imshow((res['dref'][:,:,50,:]).sum(axis = 2))
plt.show()
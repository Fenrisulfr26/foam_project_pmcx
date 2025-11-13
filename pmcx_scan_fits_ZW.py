# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:31:34 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pmcx
from scipy.optimize import curve_fit
from cv2 import resize
import cv2


#%% sim params
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
           t_res=15e-12,      # time resolution of the camera
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

#%% load the data

# IRF = np.load(r'../plastock_foam_light\XL_measurement_061224/IRF_gain0.7_timebin15ps_2000bins_061224.npy')[:1000]
IRF = np.load(r'IRF_gain0.7_timebin15ps_2000bins_061224.npy')[:1000]
IRF = IRF - IRF[:100].mean()
IRF = IRF[100:]
IRF = IRF/IRF.max()
IRF = IRF[:params['bins']]

irf_peak = np.argmax(IRF)-np.round(((params['slab_1']+params['slab_2'])*1e-2/const.c)/params['t_res'])
irf_peak = int(irf_peak)

noabs = np.load(r'noabs1_gain0.7_5x5cm_31x31points_pol50deg_expo1.0sec_binWidth15ps_binNum2000_061224.npy')[:,:,:1000]
noabs = noabs[:,:,100+irf_peak:100+irf_peak+params['bins']]
X,Y = np.meshgrid([0,15,30], [0,15,30])
no_abs = noabs[X,Y]

#%% downsample the data
time = np.linspace(0,(no_abs.shape[2]-1)*(15e-3), no_abs.shape[2])

## Downsample the irf to convolve with the results of FEM
data_irf_down = resize(IRF[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)
data_irf_down = np.squeeze(data_irf_down) # get rid of extra dimension
time_down = resize(time[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)

no_abs_down = np.zeros((3, 3, 100+1))
for i in range(3):
    for j in range(3):
        no_abs_down[i,j] = np.squeeze(resize(no_abs[i,j,:667], (1, 101), 
                                             interpolation=cv2.INTER_NEAREST_EXACT))

params['IRF'] =  data_irf_down

fit_weight = (no_abs_down/no_abs_down.max()).max(axis = 2)

#%% configure sim

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
plt.imshow(sens)

#%% functions for sims
def simfn(cfg, mu_a, mu_s):
    cfg['prop'] = [[0,0,1,1],           # background
                   [mu_a,mu_s,0,1.44]  # volume 1
                   ]
    
    # return pmcx.mcxlab(cfg)   # pmcx.mcxlab calls pmcx.run, and postprocess res['detp'] and res['traj'] raw data into dict form
    return pmcx.run(cfg)   # pmcx.mcxlab calls pmcx.run, and postprocess res['detp'] and res['traj'] raw data into dict form

def fitfn(cfg, mu_a, mu_s):
    sim_3x3 = np.zeros((3,3,101))
    for i, posy in enumerate([150, 125, 100]):
        for j, posx in enumerate([150, 125, 100]):
            cfg['srcpos'] = [posx, posy, 0]
            print(cfg['srcpos'])
            
            res = simfn(cfg, mu_a, mu_s)
            
            # retrieve the results via surface fluence
            dref_bd = res['dref'][:,:,int(vol.shape[2]-1),:] # get the boundary value
            dref_bd_rot = np.rot90(dref_bd, k=1, axes = (0,1)) # rotate the output to get correct angle
            sim_3x3[i, j] = (dref_bd_rot*sens[:,:,None]).sum((0,1))
            
            print(f'mua = {mu_a}\nmus = {mu_s}')
    
    IRF = params['IRF']
    
    sim_pad = np.zeros(((3,3,len(IRF)*4)))
        
    sim_3x3 = sim_3x3/sim_3x3.max()
    sim_pad[:,:,:len(IRF)] = sim_3x3
    F_sim = (1/(2*np.pi))*np.fft.fft(sim_pad, axis=2)
    
    IRF_pad = np.zeros((len(IRF)*4))
    IRF_pad[:int(len(IRF))] = IRF/IRF.max()
    F_IRF = (1/(2*np.pi))*np.fft.fft(IRF_pad)
    
    # time compensation since the irf in experiments is not the start time of the simulation
    # comp = np.round(((cfg['vol'].shape[2]*cfg['unitinmm']*1e-3)/const.c)/cfg['tstep']) 
    conv = np.real(np.fft.ifft(F_IRF*(F_sim)))#[int(comp):]
    conv = np.roll(conv, -np.argmax(IRF), axis=2)
    # conv = conv[29:104]/conv[29:104].max() # comment for showing full plots
    conv = conv/conv.max() # uncomment for showing full plots
    conv = conv/fit_weight[:,:,None]
    # return np.log(conv[:len(IRF)][29:104])
    return conv[:,:,:len(IRF)].flatten() 

#%%
print('fitting..\n#######################')

y = no_abs_down
y = y/y.max()
y = y/fit_weight[:,:,None]

p, pcov = curve_fit(fitfn,
                    cfg,
                    y.flatten(),
                    p0=[0.0025883,1.68196508], 
                    # bounds=([0.0001,0.5],[0.1, 2]),
                    # epsfcn=1e-5,
                    ftol=1e-5,
                    method = 'trf'
                    )
print('..fitted\n#######################')
print(f'p = {p}')

fit_sim = fitfn(cfg,*p)

# fit_sim = fitfn(cfg,0.0025883,1.68196508)
# fit_sim = fitfn(cfg, -0.0015 ,0.4969) 250907 fitting results array([-0.00154677,  0.4969274 ])
# 0.0022 1.6257 250922 fitting results
# 0.0019 1.4800 250923 fitting results

#---------250930 results default method---------
# mua = 0.0022764337085309653
# mus = 1.6521532716846619

#---------250930 results trf method---------
# mua = 0.002408133331224196
# mus = 1.7098831994700054

#%% 
# cfg['nphoton'] = 1e7
# fit_sim = fitfn(cfg,0.0019,1.4800)

# plt.figure()
# plt.imshow(mcx_res['dref'].sum(axis = (2,3)))
# plt.show()


#%%
y = no_abs_down
y = y/y.max()
plt.figure()
# plt.title(r'mu_a = 0.0019,mu_s = 1.48')
fit_sim = np.reshape(fit_sim,y.shape)
for i in range(3):
    for j in range(3):
        ax = plt.subplot(3,3,3*i+j+1)
        plt.plot(y[i,j,:], label='data')
        plt.plot(fit_sim[i,j,:]*fit_weight[i,j], label='fit')
        plt.ylim([0,1])
plt.legend()
plt.show()

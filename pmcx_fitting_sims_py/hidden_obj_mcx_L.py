# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:50:49 2025

@author: zhiguan wang
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pmcx
from scipy.optimize import curve_fit
from cv2 import resize
import cv2
import os
from tqdm import tqdm
from scipy.signal import fftconvolve

#%% read and process obj image, note the obj image is capture from the galvo view
data_fold = r'F:\OneDrive\UK_projects_local\project in UK 2024\diffuse_experiment241021\zhiguan_data\XL_measurement_061224'
obj_img_path = os.path.join(data_fold, 'L_061224.bmp')
obj_img = plt.imread(obj_img_path)

#---------------get the obj ROI defined by the nine holes-----------
obj_img_10x10 = obj_img[360:720, 827:1188]
obj_img_10x10 = obj_img_10x10/obj_img_10x10.max()
obj_img_10x10[obj_img_10x10<0.5] = 0   # keep the obj as zero,in MCX a photon is terminated when moving from a non-zero to zero voxel.
obj_img_10x10[obj_img_10x10>=0.5] = 1

#---------------find the centre from the top, bottom and side of the image edges (10x10)cm----------
pix_ratio = np.round(np.mean([obj_img_10x10.shape[0], obj_img_10x10.shape[1]])/10)
from_top = np.round((1.75 + 2.5)*pix_ratio).astype('int') # center pixel to top edge in obj_img_10x10 measured in pixel scale
from_bot = np.round((10 - 1.75 - 2.5)*pix_ratio).astype('int') # center pixel to bottom edge in obj_img_10x10 measured in pixel scale
from_side = np.round(5*pix_ratio).astype('int') # center pixel to side edge in obj_img_10x10 measured in pixel scale
pix_4cm = np.round(4*pix_ratio).astype('int') # choose border 4cm from centre for a FOV=(8x8)cm
pix_2dot5cm = np.round(2.5*pix_ratio).astype('int') # choose border 4cm from centre for a FOV=(8x8)cm

#---------------crop required size--------------
obj_img_5x5 = obj_img_10x10[from_top - pix_2dot5cm:from_top + pix_2dot5cm,
                            from_side -pix_2dot5cm:from_side + pix_2dot5cm]

obj_img_8x8 = obj_img_10x10[from_top - pix_4cm:from_top + pix_4cm,
                            from_side -pix_4cm:from_side + pix_4cm]

#-------------------show the obj img-------------------
# plt.figure()
# plt.imshow(obj_img)
# plt.show()

# plt.figure()
# plt.imshow(obj_img_10x10)
# plt.show()

# plt.figure()
# plt.imshow(obj_img_5x5)
# plt.show()

# plt.figure()
# plt.imshow(obj_img_8x8)
# plt.show()

#%% ------------------- prepare MCX cfg---------------------

#--------------note the MCX output is from the PMT view-------------
obj_in_mcx = cv2.resize(obj_img_5x5, (50,50), interpolation=cv2.INTER_NEAREST)
obj_in_mcx_PMT = np.flip(obj_in_mcx,1) # flip the obj to make sure that the obj is form the PMT view

obj_in_mcx_PMT_padding = cv2.copyMakeBorder(
    obj_in_mcx_PMT, 
    100, 100, 100, 100,  # the padding range form four sides
    cv2.BORDER_CONSTANT, 
    value=1  # the padding value
)

obj_in_mcx_PMT_padding = np.uint8(obj_in_mcx_PMT_padding);

# plt.figure()
# plt.imshow(obj_in_mcx_PMT_padding)
# plt.imshow(obj_in_mcx_PMT)
# plt.show()

# ------------creating the MCX volume--------------
thickness = 50 # mm
unitinmm = 1
t_vox = int(thickness/unitinmm)
vol = np.ones([250,250,t_vox+1],dtype='uint8')
vol[:,:,-1] = 0
vol[:,:,24] = np.rot90(obj_in_mcx_PMT_padding,k = -1,axes = (0,1)) # rotate the MCX to make sure shape is correct

#--------------MCX cfg------------------
try:
    cfg.clear()
except NameError:
    pass  
    
cfg = {'nphoton': 1e7, 
       'vol': vol, 
       'tstart':0, 
        'tend':10.1e-9, 
        'tstep':0.1e-9, # 100 ps resolution
       'srcpos': [10,125,0], 
       'srcdir':[0,0,1],
       'unitinmm':unitinmm,
       # 'detpos': [125, 125, t_vox, 4], #radius approx 1mm (4vox) (fibre ~2mm from sample)
       'issrcfrom0':1,
       'issavedet':1,
       'issaveref':1
       }


cfg['prop'] = [[0,0,1,1],           # background
               [0.0019, 1.4800,0,1.44], # volume 1
               ]


# plt.figure()
# plt.imshow(dref_bd_rot.sum(axis = 2))
# plt.show()

#-----------------prepare the sensitibity map-------------
sems_path = os.path.join(data_fold,r'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy')
sensitivity = np.load(sems_path).sum(2)
sensitivity = sensitivity/sensitivity.max()
sensitivity = np.fliplr(sensitivity)
sensitivity = resize(sensitivity, (80,80), interpolation=cv2.INTER_LINEAR)
sens_pad = np.zeros((250,250))
sens_pad[125-40:125+40, 125-40:125+40] = sensitivity

# plt.figure()
# plt.imshow(sens_pad)
# plt.show()


#-----------------prepare the IRF---------------------
IRF_path = os.path.join(data_fold,r'IRF_gain0.7_timebin15ps_2000bins_061224.npy')
IRF = np.load(IRF_path)[:1000]
IRF_down = resize(IRF[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)
IRF_down = np.squeeze(IRF_down)
# plt.plot(IRF_down)
# plt.plot(IRF)
#%% -----------------scan the points to get measurements------------

sim_mea = np.zeros((31,31,101));

src_x_positions = np.linspace(125+25,125-25,31) # from PMT view, from right to left
src_y_positions = np.linspace(125+25,125-25,31) # from PMT view, from up to down

# --------------scan the laser point------------
# ---------i is row, which is Y, j is col(X)
for i, pos_y in tqdm(enumerate(src_y_positions)):
    for j, pos_x in tqdm(enumerate(src_x_positions)):
        cfg['srcpos'] = [pos_x, pos_y, 0]
        res = pmcx.run(cfg);
        dref_bd = res['dref'][:,:,int(vol.shape[2]-1),:] # get the boundary value
        dref_bd_rot = np.rot90(dref_bd, k=1, axes = (0,1)) # rotate to get correct angle
        sim_mea[i,j,:] = (dref_bd_rot*sens_pad[:,:,None]).sum((0,1)) # apply the sensitivity map
            
sim_mea = np.apply_along_axis(lambda m: fftconvolve(m, IRF_down, mode='full'), axis=2, arr=sim_mea)[:,:,:len(IRF_down)]

sim_mea = sim_mea/sim_mea.max()
    
mea = np.load(os.path.join(data_fold,r'L_gain0.7_5x5cm_31x31points_pol50deg_expo0.5sec_binWidth15ps_binNum2000_061224.npy'));

mea_down = np.zeros((31, 31, 100+1))
for i in range(31):
    for j in range(31):
        mea_down[i,j,:] = np.squeeze(resize(mea[i,j,:667], (1, 101), 
                                             interpolation=cv2.INTER_NEAREST_EXACT))
mea_down = mea_down/mea_down.max()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(sim_mea.sum(axis = 2),cmap='jet')
plt.title('MCX simulation')
plt.subplot(1,2,2)
plt.imshow(mea.sum(axis = 2),cmap = 'jet')
plt.title('measurements')
plt.show()


np.save('sim_mea_obj_L.npy',sim_mea)

#%% check if the histograms are good

plt.figure()

for i, index_i in enumerate([0, 15, 30]):
    for j,index_j in enumerate([0, 15, 30]):
        plt.subplot(3,3,3*i+j+1)
        # plt.plot(y[i,j,:], label='data')
        plt.plot(sim_mea[index_i,index_j,:], label='fit')
        plt.plot(mea_down[index_i,index_j,:], label='ground truth')
        # plt.plot(conv_simp[i,j,:len(IRF)], label='fit')
        plt.ylim([0,1.1])
        plt.legend()
plt.show()

# %% only compare the 3x3 results

cfg['prop'] = [[0,0,1,1],           # background
               [0.0019, 1.48,0,1.44], # volume 1
               ]


sim_mea_3x3 = np.zeros((3,3,101));

src_x_positions_3x3 = np.linspace(125+25,125-25,3) # from PMT view, from right to left
src_y_positions_3x3 = np.linspace(125+25,125-25,3) # from PMT view, from up to down

# --------------scan the laser point------------
# ---------i is row, which is Y, j is col(X)
for i, pos_y in tqdm(enumerate(src_y_positions_3x3)):
    for j, pos_x in tqdm(enumerate(src_x_positions_3x3)):
        cfg['srcpos'] = [pos_x, pos_y, 0]
        res = pmcx.run(cfg);
        dref_bd = res['dref'][:,:,int(vol.shape[2]-1),:] # get the boundary value
        dref_bd_rot = np.rot90(dref_bd, k=1, axes = (0,1)) # rotate to get correct angle
        sim_mea[i,j,:] = (dref_bd_rot*sens_pad[:,:,None]).sum((0,1)) # apply the sensitivity map
            
sim_mea = np.apply_along_axis(lambda m: fftconvolve(m, IRF_down, mode='full'), axis=2, arr=sim_mea)[:,:,:len(IRF_down)]

sim_mea = sim_mea/sim_mea.max()
    
mea = np.load(os.path.join(data_fold,r'L_gain0.7_5x5cm_31x31points_pol50deg_expo0.5sec_binWidth15ps_binNum2000_061224.npy'));

mea_down3x3 = np.zeros((3, 3, 101))

for i, index_i in enumerate([0, 15, 30]):
    for j,index_j in enumerate([0, 15, 30]):
        mea_down3x3[i,j,:] = np.squeeze(resize(mea[index_i,index_j,:667], (1, 101), 
                                             interpolation=cv2.INTER_NEAREST_EXACT))

mea_down3x3 = mea_down3x3/mea_down3x3.max()


plt.figure()

for i, index_i in enumerate([0, 15, 30]):
    for j,index_j in enumerate([0, 15, 30]):
        plt.subplot(3,3,3*i+j+1)
        plt.plot(sim_mea[i,j,:], label='fit')
        plt.plot(mea_down3x3[i,j,:], label='ground truth')
        plt.ylim([0,1.1])
        plt.legend()
plt.show()
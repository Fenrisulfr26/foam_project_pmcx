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
from torchvision import datasets, transforms
import torchvision
import torch
import collections
from torch.utils.data import DataLoader,Subset
import pickle

# %% prepare the dataset
ROOT = "F:\OneDrive\my_tronscan\data"  # root fold for the data       
SPLIT = "byclass"       
NUM_TOTAL = 5000      
RANDOM_SEED = 42       
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.ToTensor()
])

train_ds = datasets.EMNIST(root=ROOT, split=SPLIT, train=True, download=True, transform=transform)
test_ds  = datasets.EMNIST(root=ROOT, split=SPLIT, train=False, download=True, transform=transform)

class ConcatDatasetLike(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.len1 = len(ds1)
        self.len2 = len(ds2)
    def __len__(self):
        return self.len1 + self.len2
    def __getitem__(self, idx):
        if idx < self.len1:
            return self.ds1[idx]
        else:
            return self.ds2[idx - self.len1]

full_ds = ConcatDatasetLike(train_ds, test_ds)

# collect the index of every label and fetch the label of
label_to_indices = collections.defaultdict(list)
for idx in range(len(full_ds)):
    _, label = full_ds[idx]
    # EMNIST byclass: labels 0-61 
    if 10 <= label <= 35:
        label_to_indices[int(label)].append(idx)

num_classes = 26
base = NUM_TOTAL // num_classes            # 192
remainder = NUM_TOTAL % num_classes       # 8
per_class_target = {i: base + (1 if i < remainder else 0) for i in range(num_classes)}

# random sampling
g = torch.Generator().manual_seed(RANDOM_SEED)
selected_indices = []

for i in range(num_classes):
    label = 10 + i
    available = label_to_indices.get(label, [])
    want = per_class_target[i]
    if len(available) >= want:
        perm = torch.randperm(len(available), generator=g)
        chosen_local = [available[int(perm[j])] for j in range(want)]
        selected_indices.extend(chosen_local)
    else:
        print(f"Warning: label {label} ({chr(ord('A')+i)}) has only {len(available)} samples, requested {want}. Using all available.")
        selected_indices.extend(available)

print("Total selected:", len(selected_indices))

if len(selected_indices) > NUM_TOTAL:
    selected_indices = selected_indices[:NUM_TOTAL]
elif len(selected_indices) < NUM_TOTAL:
    print(f"Note: only {len(selected_indices)} samples were collected (<{NUM_TOTAL}). Consider lowering NUM_TOTAL or allowing imbalanced sampling.")

subset_ds = Subset(full_ds, selected_indices)

loader = DataLoader(subset_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# show numbers in each label
count_by_label = collections.Counter()
for idx in selected_indices:
    _, label = full_ds[idx]
    count_by_label[int(label)] += 1

print("Per-class counts (label -> count) for A-Z:")
for i in range(num_classes):
    lab = 10 + i
    ch = chr(ord("A") + i)
    print(f"{ch} (label {lab}): {count_by_label.get(lab,0)}")




# %% get the sensitivity map and the IRF
data_fold = r'F:\OneDrive\UK_projects_local\project in UK 2024\diffuse_experiment241021\zhiguan_data\XL_measurement_061224'

#-----------------prepare the sensitibity map-------------
sens_path = os.path.join(data_fold,r'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy')
sensitivity = np.load(sens_path).sum(2)
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
IRF_down= np.squeeze(IRF_down)
# plt.plot(IRF_down)

#%% get the obj figure and loop the simulation

sim_len = 121

try:
    sim_results
    ground_truths
except NameError:
    sim_results = np.zeros([3,3,sim_len,5000])
    ground_truths = np.zeros([50,50,5000])


src_x_positions = np.linspace(125+25,125-25,3) # from PMT view, from right to left
src_y_positions = np.linspace(125+25,125-25,3) # from PMT view, from up to down

#--------------MCX cfg------------------
try:
    cfg.clear()
except NameError:
    pass  
    
cfg = {'nphoton': 1e7, 
       'tstart':0, 
        'tend':sim_len*1e-10, 
        'tstep':0.1e-9, # 100 ps resolution
       'srcpos': [10,125,0], 
       'srcdir':[0,0,1],
       'unitinmm':1,
       # 'detpos': [125, 125, t_vox, 4], #radius approx 1mm (4vox) (fibre ~2mm from sample)
       'issrcfrom0':1,
       'issavedet':1,
       'issaveref':1
       }


cfg['prop'] = [[0,0,1,1],           # background
               [0.0019, 1.4800,0,1.44], # volume 1
               ]
    

for img_index in tqdm(range(5000)):
    print(img_index)
    img, lab = subset_ds[img_index]  
    img = np.fliplr(np.rot90(img.squeeze(),k = -1,axes = (0,1))) # change the shape to the right orientation
    img = cv2.copyMakeBorder(
        img,
        top=2, bottom=2, left=2, right=2,  
        borderType=cv2.BORDER_CONSTANT,
        value=0   
    )
    
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
    
    img[img<=0.5]=0
    img[img>0.5]=1
    img = 1-img
    
    # plt.figure()
    # plt.imshow(img.squeeze())
    # plt.show()
        
    ground_truths[:,:,img_index] = img
 
    obj_in_mcx_PMT_padding = cv2.copyMakeBorder(
        img, 
        100, 100, 100, 100,  # the padding range form four sides
        cv2.BORDER_CONSTANT, 
        value=1  # the padding value
    )
    
    obj_in_mcx_PMT_padding = np.uint8(obj_in_mcx_PMT_padding);
    
    
    # ------------creating the MCX volume--------------
    thickness = 50 # mm
    unitinmm = 1
    t_vox = int(thickness/unitinmm)
    vol = np.ones([250,250,t_vox+1],dtype='uint8')
    vol[:,:,-1] = 0
    vol[:,:,24] = np.rot90(obj_in_mcx_PMT_padding,k = -1,axes = (0,1)) # rotate the MCX to make sure shape is correct
    
    cfg['vol'] = vol

    sim_mea = np.zeros((3,3,sim_len));
    
    # --------------scan the laser point------------
    # ---------i is row, which is Y, j is col(X)
    for i, pos_y in enumerate(src_y_positions):
        for j, pos_x in enumerate(src_x_positions):
            cfg['srcpos'] = [pos_x, pos_y, 0]
            res = pmcx.run(cfg);
            dref_bd = res['dref'][:,:,int(vol.shape[2]-1),:] # get the boundary value
            dref_bd_rot = np.rot90(dref_bd, k=1, axes = (0,1)) # rotate to get correct angle
            sim_mea[i,j,:] = (dref_bd_rot*sens_pad[:,:,None]).sum((0,1)) # apply the sensitivity map
                
    sim_mea = np.apply_along_axis(lambda m: fftconvolve(m, IRF_down, mode='full'), axis=2, arr=sim_mea)[:,:,:sim_len]
    
    sim_mea = sim_mea/sim_mea.max()
    
    sim_results[:,:,:,img_index] = sim_mea
    
    np.save('batch_sim_results.npy',sim_results)
    np.save('groundtruths.npy',ground_truths)

# %% check the results
ground_truths = np.load('groundtruths.npy')
sim_results = np.load('batch_sim_results.npy')

# plt.figure()
# plt.plot(sim_results[2,1,:,1000])
# plt.show()

plt.figure()
plt.imshow(sim_results[:,:,:,200].sum(axis=2))

plt.figure()
plt.imshow(ground_truths[:,:,1])
plt.show()

img1, lab = subset_ds[1]  
plt.figure()
plt.imshow(img1.squeeze())

plt.figure()
plt.plot(sim_results[1,1,:,:].sum(axis = 0))
plt.show()

plt.figure()
plt.plot(sim_results[1,1,:,1000])
plt.show()
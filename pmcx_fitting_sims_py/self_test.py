# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:55:08 2025

use MCX data as the ground truth to see if the optimization works good

@author: zhiguan wang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pmcx
from scipy.optimize import curve_fit
from cv2 import resize
import cv2
from scipy.signal import fftconvolve

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
#%% generate data from analytical model
"""
Spatio-temporal diffuse light field at rear surface of a slab (diffusion approximation).
- Slab size: XxY = 250 x 250 mm (we evaluate a grid on rear surface z = L)
- Thickness: L = 50 mm
- Point source on front surface z0 (user can scan x0,y0)
- Uses diffusion Green's function with image sources to satisfy extrapolated-boundary conditions
- Outputs time-resolved fluence (relative) on rear surface and plotting utilities

Notes:
- This is an approximate diffusion-model solver (fast). For high accuracy use Monte Carlo.
- Units: mm and seconds.
"""


# -----------------------------
# Physical / optical parameters
# -----------------------------
L = 50.0          # slab thickness (mm)
X = 250.0         # lateral size in x (mm)
Y = 250.0         # lateral size in y (mm)
nx = 250         # grid points in x
ny = 250          # grid points in y

# Typical optical properties (tissue-like, NIR)
mu_a = 0.01       # absorption coefficient (mm^-1)
mu_sp = 1.0       # reduced scattering coefficient mu_s' (mm^-1)
n = 1.4           # refractive index of medium

# Derived diffusion parameters
D = 1.0 / (3.0 * (mu_a + mu_sp))   # diffusion coefficient (mm)
c_mm_s = 3e11                      # speed of light in mm/s
v = c_mm_s / n                     # speed of light in medium (mm/s)
# Effective diffusion constant in time-domain kernels: D_time = D * v
D_time = D * v

print(f"D = {D:.4e} mm, v = {v:.4e} mm/s, D_time = {D_time:.4e} mm^2/s")

# Extrapolated boundary length z_b (approximate)
# Using Fresnel reflectance approximation to compute A -> z_b = 2*A*D
# Approximate internal reflection coefficient for diffuse light (Jenkins & White / Farrell)
# Use interpolation formula for planar boundary:
def diffuse_reflection_factor(n_rel):
    # approximate internal reflection for diffuse boundary (from literature approximations)
    # n_rel = n_medium / n_outside (outside assumed air ~1.0)
    n = n_rel
    r0 = -1.440 / (n**2) + 0.710 / n + 0.668 + 0.0636 * n  # empirical approx (valid near 1-2)
    # ensure between 0 and 1
    return np.clip(r0, 0.0, 0.999)
R = diffuse_reflection_factor(n)   # effective reflection
A = (1 + R) / (1 - R)
z_b = 2 * A * D
print(f"Estimated internal reflectance R={R:.4f}, A={A:.4f}, extrapolation length z_b={z_b:.4f} mm")

# Extrapolated boundary positions at z = -z_b and z = L + z_b
z_extrap_neg = -z_b
z_extrap_pos = L + z_b

# -----------------------------
# Time axis (choose relevant window)
# -----------------------------
# choose time window: start from t_min to t_max (seconds)
# diffusion arrival times scale roughly as t ~ L^2 / (D_time)
t_scaling = (L**2) / D_time
# choose times from small fraction to a few times scaling
t_min = 0.1e-9
t_max = 100.1e-9
nt = 101
t = np.linspace((t_min), (t_max), nt)  # log-spaced times (s)
print(f"Time window: {t_min:.3e} s .. {t_max:.3e} s (nt={nt})")

# -----------------------------
# Spatial grid on rear surface z = L
# -----------------------------
xs = np.linspace(-X/2, X/2, nx)
ys = np.linspace(-Y/2, Y/2, ny)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
rr2 = XX**2 + YY**2

# -----------------------------
# Green's function (image source) implementation
# -----------------------------
# We implement the time-dependent point-source Green's function in an infinite medium:
# G_inf(r, z, t; z0) ~ (1 / ( (4*pi*D_time*t)^(3/2) )) * exp(-mu_a * v * t) * exp( - (rho^2 + (z-z0)^2) / (4*D_time*t) )
# For a slab with extrapolated boundaries at -z_b and L+z_b, we sum image sources at:
# z_m = 2*m*(L + 2*z_b) + s * z0, with s = ±1, and alternate signs to enforce Dirichlet at extrapolated planes.
# This is a classical approximate construction — see Patterson/Chance style solutions.
#
# We'll sum images for m in [-M, M] and s in {+1, -1} with appropriate sign.
#
# Source strength normalization is chosen so resulting fluence is in arbitrary relative units.

def greens_time_contribution(rho2, z_target, z_src, t_scalar, D_time, mu_a, v):
    """
    Contribution of a single (instantaneous) point source image located at z_src
    to fluence at (rho^2, z_target) at time t_scalar.
    Returns relative fluence (no absolute photon number normalization).
    """
    if t_scalar <= 0:
        return 0.0
    coef = 1.0 / ((4.0 * np.pi * D_time * t_scalar) ** 1.5)
    expo = np.exp(-mu_a * v * t_scalar)
    dist2 = rho2 + (z_target - z_src)**2
    gauss = np.exp(-dist2 / (4.0 * D_time * t_scalar))
    return coef * expo * gauss

# -----------------------------
# Main simulation function
# -----------------------------
def compute_rear_surface_time_series(x0=0.0, y0=0.0, z0=0.0, times=t, verbose=False):
    """
    Compute the time-resolved fluence on rear surface z = L for a point source at (x0,y0,z0)
    x0,y0 in mm (on front surface plane), z0 depth in mm (0 ~ at front surface).
    times: array of times in seconds
    M: number of image periods each side (increase for better accuracy; cost ~ (2M+1))
    Returns: fluence_time: shape (nt, ny, nx)
    """
    nt = len(times)
    fluence_time = np.zeros((nt, ny, nx), dtype=np.float64)
    # lateral offset between source position and grid: assume source at (x0,y0)
    # Our precomputed XX,YY are coordinates relative to slab center; adjust
    dx = XX - x0
    dy = YY - y0
    rho2_grid = dx**2 + dy**2
    for i in range(nt):
        if verbose and (i % max(1, nt//10) == 0):
            print(f"Computing t index {i+1}/{nt}, t={times[i]:.3e} s")
        fluence_time[i] = greens_time_contribution(rho2_grid, z_target=L, z_src=z0, t_scalar=times[i],
                                         D_time=D_time, mu_a=mu_a, v=v)
    # normalize to max=1 for convenience (relative fluence)
    # but keep full time evolution scaling (we normalize per max across all times)
    maxval = np.max(fluence_time)
    if maxval > 0:
        fluence_time /= maxval
    return fluence_time


sensitivity = np.load('sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy').sum(2)
sensitivity = sensitivity/sensitivity.max()
sensitivity = np.fliplr(sensitivity)
sensitivity = resize(sensitivity, (80,80), interpolation=cv2.INTER_LINEAR)
sens = np.zeros((250,250))
sens[125-40:125+40, 125-40:125+40] = sensitivity

anly_3x3 = np.zeros([3,3,101])

for i,x_src in enumerate([-25, 0, 25]):
    for j,y_src in enumerate([-25, 0, 25]):
        z0 = 1e-3   # put source just inside front surface to avoid singularity (mm)
        print("Starting compute (this may take ~tens of seconds depending on nt,nx,ny,M)...")
        flu_ts = compute_rear_surface_time_series(x0=x_src, y0=y_src, z0=z0, times=t, verbose=False)
        anly_3x3[i,j,:] = (flu_ts*sens).sum(axis = (1,2))


anly_3x3 = anly_3x3/anly_3x3.max()
anly_3x3_c = np.apply_along_axis(lambda m: fftconvolve(m, data_irf_down, mode='full'), axis=2, arr=anly_3x3)
anly_3x3_c = anly_3x3_c/anly_3x3_c.max()
anly_3x3_c = anly_3x3_c[:,:,:len(data_irf_down)]

plt.figure()
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        plt.plot(anly_3x3[i,j,:]/anly_3x3.max(), label='anly')
        plt.plot(anly_3x3_c[i,j,:], label='anly')
        plt.ylim([0,1])
plt.legend()
plt.show()

plt.figure()
plt.imshow(anly_3x3_c.sum(axis = 2))

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
    conv_simp = conv_simp/weight[:,:,None]
    # plt.figure()
    # plt.imshow((conv_simp[:,:,:len(IRF)]).sum(axis = 2))
    
    # sim_3x3 = sim_3x3/sim_3x3.max()
    
    # for i in range(3):
    #     for j in range(3):
    #         plt.subplot(3,3,3*i+j+1)
    #         plt.plot(anly_3x3[i,j,:], label='data')
    #         plt.plot(sim_3x3[i,j,:], label='fit')
    #         # plt.plot(conv_simp[i,j,:len(IRF)], label='fit')
    #         plt.ylim([0,1])
    #         plt.title(', '.join(f'{x:.4f}' for x in p))
    # plt.legend()
    # plt.show()
    
    return conv_simp.flatten() 

#%%

# generate ref data as the fitting target

sim_3x3 = np.zeros((3,3,101))
for i, posx in enumerate([100, 125, 150]):
    for j, posy in enumerate([100, 125, 150]):
         cfg['srcpos'] = [posx, posy, 0]
         print(cfg['srcpos'])
         
         res = simfn(cfg, 0.01, 1)
         
         # retrieve the results via surface fluence
         sim = (res['dref'][:,:,int(vol.shape[2]-1),:]*sens[:,:,None]).sum((0,1))
         sim_3x3[i, j] = sim
          
IRF = params['IRF']
grd_tru = np.apply_along_axis(lambda m: fftconvolve(m, IRF, mode='full'), axis=2, arr=sim_3x3)[:,:,:len(IRF)]
grd_tru = grd_tru/grd_tru.max()
grd_tru = grd_tru[:,:,:101]
weight = grd_tru.max(axis = 2)


print('fitting..\n#######################')

p, pcov = curve_fit(
    fitfn, 
    cfg, 
    (grd_tru/weight[:,:,None]).flatten(), 
    p0=[0.015,1.2],
    # bounds=([0.001,1.0],[0.005, 1.8]),
    ftol=1e-5, 
    method='trf'
   )

print('..fitted\n#######################')
print(f'p = {p}')

fit_sim = fitfn(cfg,*p)

# fit_sim = fitfn(cfg,0.0025883,1.68196508)
# fit_sim = fitfn(cfg, -0.0015 ,0.4969) 250907 fitting results array([-0.00154677,  0.4969274 ])


#%% show the results
y = anly_3x3_c
y = y/y.max()


plt.figure()
fit_sim = np.reshape(fit_sim,[3,3,101])
# array([0.02073704, 1.90952645])
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        # plt.plot(y[i,j,:], label='data')
        plt.plot(fit_sim[i,j,:], label='fit')
        plt.plot((grd_tru/weight[:,:,None])[i,j,:], label='ground truth')
        # plt.plot(conv_simp[i,j,:len(IRF)], label='fit')
        plt.ylim([0,1.2])
        # plt.title(', '.join(f'{x:.4f}' for x in p))
plt.legend()
plt.show()


plt.figure()
fit_sim = np.reshape(fit_sim,[3,3,101])
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        # plt.plot(y[i,j,:], label='data')
        plt.plot(fit_sim[i,j,:]*weight[i,j], label='fit')
        plt.plot((grd_tru)[i,j,:], label='ground truth')
        # plt.plot(conv_simp[i,j,:len(IRF)], label='fit')
        plt.ylim([0,1.2])
        # plt.title(', '.join(f'{x:.4f}' for x in p))
plt.legend()
plt.show()
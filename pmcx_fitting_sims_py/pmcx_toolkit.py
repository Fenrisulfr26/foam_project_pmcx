import numpy as np
import pmcx
from cv2 import resize
import cv2
import os
from scipy.signal import fftconvolve
import warnings

class pmcx_obj(object):
    
    def __init__(self, obj):
        self.obj = obj

    def pmcx_sim(self):

        if self.obj.shape[:2] != (50, 50):
            warnings.warn(f"Input image shape {self.obj.shape[:2]} is not (50, 50). It will be resized.")
            self.obj = cv2.resize(self.obj, (50, 50), interpolation=cv2.INTER_NEAREST)
        else:
            self.obj = self.obj


        data_fold = r'F:\OneDrive\UK_projects_local\project in UK 2024\diffuse_experiment241021\zhiguan_data\XL_measurement_061224'

        #-----------------prepare the sensitibity map for the MCX-------------
        sems_path = os.path.join(data_fold,r'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy')
        sensitivity_mcx = np.load(sems_path).sum(2)
        sensitivity_mcx = sensitivity_mcx/sensitivity_mcx.max()
        sensitivity_mcx = np.fliplr(sensitivity_mcx)
        sensitivity_mcx = resize(sensitivity_mcx, (80,80), interpolation=cv2.INTER_LINEAR)
        sens_pad = np.zeros((250,250))
        sens_pad[125-40:125+40, 125-40:125+40] = sensitivity_mcx

        #-----------------prepare the IRF for the MCX---------------------
        IRF_path = os.path.join(data_fold,r'IRF_gain0.7_timebin15ps_2000bins_061224.npy')
        IRF = np.load(IRF_path)[:1000]
        # IRF_down = resize(IRF[:667], (1, 101), interpolation=cv2.INTER_NEAREST_EXACT)
        IRF_down = resize(IRF[:np.round(121*100/15).astype('int')], (1, 121), interpolation=cv2.INTER_NEAREST_EXACT)
        IRF_down= np.squeeze(IRF_down)
        # plt.plot(IRF_down)

        #===============prepare the sensitivity map for the analytical model===================t
        sensitivity_path = os.path.join(data_fold, r'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy')
        sensitivity = np.load(sensitivity_path).sum(2)
        # %matplotlib qt
        # plt.figure(),plt.imshow(sensitivity, cmap='hot')
        sensitivity = sensitivity - 0.3*sensitivity.max()
        # plt.figure(),plt.imshow(sensitivity, cmap='hot')
        sensitivity[sensitivity<0] = 0
        sensitivity[sensitivity>0] = 1
        sensitivity = sensitivity/sensitivity.max()
        sensitivity = np.fliplr(sensitivity)

        #--------------note the MCX output is from the PMT view-------------
        obj_in_mcx_PMT =  cv2.resize(self.obj, (50, 50), interpolation=cv2.INTER_NEAREST)

        obj_in_mcx_PMT_padding = cv2.copyMakeBorder(
            obj_in_mcx_PMT, 
            100, 100, 100, 100,  # the padding range form four sides
            cv2.BORDER_CONSTANT, 
            value=1  # the padding value
        )

        obj_in_mcx_PMT_padding = np.uint8(obj_in_mcx_PMT_padding)

        # ------------creating the MCX volume--------------
        thickness = 50 # mm
        unitinmm = 1
        t_vox = int(thickness/unitinmm)
        vol = np.ones([250,250,t_vox+1],dtype='uint8')
        vol[:,:,-1] = 0
        vol[:,:,24] = np.rot90(obj_in_mcx_PMT_padding,k = -1,axes = (0,1)) # rotate the input OBJ in the MCX to make sure shape is correct


        # ------------creating the MCX volume--------------
        thickness = 50 # mm
        unitinmm = 1
        t_vox = int(thickness/unitinmm)
        vol = np.ones([250,250,t_vox+1],dtype='uint8')
        vol[:,:,-1] = 0
        vol[:,:,24] = np.rot90(obj_in_mcx_PMT_padding,k = -1,axes = (0,1)) # rotate the input OBJ in the MCX to make sure shape is correct

        #--------------MCX cfg------------------
        try:
            cfg.clear()
        except NameError:
            pass  

        cfg = {'nphoton': 1e7, 
                'vol': vol, 
                'tstart':0, 
                'tend':12.1e-9, # 10.1
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

        MCX_sim_ds = np.zeros((3,3,121))

        src_x_positions = np.linspace(125+25,125-25,MCX_sim_ds.shape[0]) # from PMT view, from right to left
        src_y_positions = np.linspace(125+25,125-25,MCX_sim_ds.shape[0]) # from PMT view, from up to down

        # --------------scan the laser point------------
        # ---------i is row, which is Y, j is col(X)
        for i, pos_y in (enumerate(src_y_positions)):
            for j, pos_x in (enumerate(src_x_positions)):
                cfg['srcpos'] = [pos_x, pos_y, 0]
                res = pmcx.run(cfg)
                dref_bd = res['dref'][:,:,int(vol.shape[2]-1),:] # get the boundary value
                dref_bd_rot = np.rot90(dref_bd, k=1, axes = (0,1)) # rotate the output to get correct angle
                MCX_sim_ds[i,j,:] = (dref_bd_rot*sens_pad[:,:,None]).sum((0,1)) # apply the sensitivity map
                    
        MCX_sim_ds = np.apply_along_axis(lambda m: fftconvolve(m, IRF_down, mode='full'), axis=2, arr=MCX_sim_ds)[:,:,:len(IRF_down)]
        MCX_sim_ds = MCX_sim_ds/MCX_sim_ds.max()
        return MCX_sim_ds



import numpy as np
import matplotlib.pyplot as plt
import pmcx
from cv2 import resize
import cv2
import os
from tqdm import tqdm
from scipy.signal import fftconvolve
import gc  # <--- [新增] 引入垃圾回收模块

class MCXFoamSimulator:
    def __init__(self, data_fold, unit_mm=0.5, foam_size_mm=126):
        """
        初始化仿真器，设置基本参数和路径。
        """
        self.data_fold = data_fold
        self.mcx_unitinmm = unit_mm
        self.foam_size = foam_size_mm
        
        # 内部状态变量
        self.obj_processed = None
        self.sensitivity_pad = None
        self.irf_down = None
        self.vol = None
        self.cfg = {}
        self.results = None 

    def load_and_process_resources(self, obj_filename, sens_filename, irf_filename):
        """
        加载图像、灵敏度图和IRF，并执行所有的预处理。
        """
        # 1. 处理 Object Image
        obj_path = os.path.join(self.data_fold, obj_filename)
        obj_img = plt.imread(obj_path)
        
        obj_img_10x10 = obj_img[363:719, 799:1156]
        obj_img_10x10 = obj_img_10x10 / obj_img_10x10.max()
        obj_img_10x10[obj_img_10x10 < 0.5] = 0
        obj_img_10x10[obj_img_10x10 >= 0.5] = 1

        pix_ratio = np.round(np.mean([obj_img_10x10.shape[0], obj_img_10x10.shape[1]])/10)
        from_top = np.round((1.75 + 2.5)*pix_ratio).astype('int')
        from_side = np.round(5*pix_ratio).astype('int')
        pix_2dot5cm = np.round(2.5*pix_ratio).astype('int')

        obj_img_5x5 = obj_img_10x10[from_top - pix_2dot5cm:from_top + pix_2dot5cm,
                                    from_side -pix_2dot5cm:from_side + pix_2dot5cm]

        target_dim = int(np.round(50/self.mcx_unitinmm))
        obj_in_mcx = cv2.resize(obj_img_5x5, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)
        obj_in_mcx_PMT = np.flip(obj_in_mcx, 1)

        padding_pixels = np.round((self.foam_size - 50)/2/self.mcx_unitinmm).astype('int')
        self.obj_processed = cv2.copyMakeBorder(
            obj_in_mcx_PMT, 
            padding_pixels, padding_pixels, padding_pixels, padding_pixels,
            cv2.BORDER_CONSTANT, value=1
        ).astype('uint8')

        # 2. 处理 Sensitivity Map
        sens_path = os.path.join(self.data_fold, sens_filename)
        sensitivity = np.load(sens_path).sum(2)
        sensitivity = sensitivity / sensitivity.max()
        sensitivity = np.fliplr(sensitivity)
        
        sens_dim = int(np.round(80/self.mcx_unitinmm))
        sensitivity = resize(sensitivity, (sens_dim, sens_dim), interpolation=cv2.INTER_LINEAR)
        
        sens_padding = int((self.foam_size - 80)/2/self.mcx_unitinmm)
        self.sensitivity_pad = cv2.copyMakeBorder(
            sensitivity, 
            sens_padding, sens_padding, sens_padding, sens_padding,
            cv2.BORDER_CONSTANT, value=0
        )

        # 3. 处理 IRF
        irf_path = os.path.join(self.data_fold, irf_filename)
        IRF = np.load(irf_path)[:1000]
        target_len = 121
        samples_needed = np.round(target_len * 100 / 15).astype('int') 
        IRF_down = resize(IRF[:samples_needed], (1, target_len), interpolation=cv2.INTER_NEAREST_EXACT)
        IRF_down = np.squeeze(IRF_down)
        self.irf_down = IRF_down / IRF_down.max()

        print(f"Resources loaded. Volume shape: {self.obj_processed.shape}, IRF len: {len(self.irf_down)}")

    def build_volume(self, thickness_mm=50):
        """构建 3D MCX 体积。"""
        if self.obj_processed is None:
            raise ValueError("Please call load_and_process_resources first.")

        t_vox = int(thickness_mm / self.mcx_unitinmm)
        xy_dim = int(np.round(self.foam_size / self.mcx_unitinmm))
        
        # 显式清理旧的 volume 防止内存累积
        if self.vol is not None:
            del self.vol
            gc.collect()

        self.vol = np.ones([xy_dim, xy_dim, t_vox + 1], dtype='uint8')
        self.vol[:, :, -1] = 0 
        obj_layer_idx = int(self.vol.shape[2]/2) - 1
        self.vol[:, :, obj_layer_idx] = np.rot90(self.obj_processed, k=-1, axes=(0, 1))

    def release_memory(self):
        """
        [新增] 手动释放所有大块内存的方法
        """
        if self.vol is not None:
            del self.vol
            self.vol = None
        if self.results is not None:
            del self.results
            self.results = None
        # 清除 cfg 中的引用
        self.cfg = {}
        gc.collect()
        print("Memory released.")

    def run_grid_scan(self, mu_a, mu_s, grid_shape=(3,3), nphoton=1e7, 
                      save_raw=False, save_dir='raw_data_output'):
        """
        运行网格扫描模拟，包含内存自动释放机制。
        """
        if self.vol is None:
            self.build_volume()
            
        if save_raw:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Raw data will be saved to: {os.path.abspath(save_dir)}")

        # 配置 MCX
        self.cfg = {
            'nphoton': nphoton,
            'vol': self.vol,
            'tstart': 0,
            'tend': 12.1e-9,
            'tstep': 0.1e-9,
            'unitinmm': self.mcx_unitinmm,
            'issrcfrom0': 1,
            'issaveref': 1,
            'srcdir': [0, 0, 1],
            'prop': [
                [0, 0, 1, 1],                # background
                [mu_a, mu_s, 0, 1.44],       # volume 1
            ]
        }

        # 准备输出容器
        time_bins = 121
        sim_results = np.zeros((grid_shape[0], grid_shape[1], time_bins))
        
        center_vox = self.foam_size / 2
        range_mm = 25
        
        src_x_pos = np.linspace(
            np.round((center_vox + range_mm)/self.mcx_unitinmm).astype(int),
            np.round((center_vox - range_mm)/self.mcx_unitinmm).astype(int),
            grid_shape[1]
        )
        src_y_pos = np.linspace(
            np.round((center_vox + range_mm)/self.mcx_unitinmm).astype(int),
            np.round((center_vox - range_mm)/self.mcx_unitinmm).astype(int),
            grid_shape[0]
        )

        print(f"Running simulation for mua={mu_a}, mus={mu_s}...")
        
        for i, pos_y in tqdm(enumerate(src_y_pos), total=len(src_y_pos)):
            for j, pos_x in enumerate(src_x_pos):
                try:
                    self.cfg['srcpos'] = [pos_x, pos_y, 0]
                    
                    # 1. 运行 MCX
                    # 注意：res 包含大量GPU返回的数据
                    res = pmcx.run(self.cfg)
                    
                    # 2. 获取边界数据并旋转
                    # 这里使用了切片，创建了新的内存副本
                    dref_bd = res['dref'][:,:,int(self.vol.shape[2]-1),:]
                    dref_bd_rot = np.rot90(dref_bd, k=1, axes=(0,1))
                    
                    # 保存原始数据
                    if save_raw:
                        file_name = f"dref_row{i}_col{j}_mua{mu_a}_mus{mu_s}_foamsize{self.foam_size}_resolution{self.mcx_unitinmm}.npy"
                        file_path = os.path.join(save_dir, file_name)
                        np.save(file_path, dref_bd_rot)

                    # 3. 处理数据
                    raw_curve = (dref_bd_rot * self.sensitivity_pad[:,:,None]).sum((0,1))
                    
                    if len(raw_curve) >= time_bins:
                        sim_results[i, j, :] = raw_curve[:time_bins]
                    else:
                        sim_results[i, j, :len(raw_curve)] = raw_curve

                except Exception as e:
                    print(f"Error at grid ({i},{j}): {e}")
                    # 即使报错也要尝试清理
                finally:
                    # ==========================================
                    # [关键修改] 显式释放内存
                    # ==========================================
                    if 'res' in locals():
                        del res
                    if 'dref_bd' in locals():
                        del dref_bd
                    if 'dref_bd_rot' in locals():
                        del dref_bd_rot
                    
                    # 强制 Python 进行垃圾回收，这是释放 GPU 句柄的关键
                    gc.collect() 
                    # ==========================================

        # 后处理
        print("Convolving with IRF...")
        sim_results = np.apply_along_axis(
            lambda m: fftconvolve(m, self.irf_down, mode='full'), 
            axis=2, 
            arr=sim_results
        )[:, :, :len(self.irf_down)]
        
        if sim_results.max() > 0:
            sim_results = sim_results / sim_results.max()
            
        self.results = sim_results
        return sim_results
    
    def plot_comparison(self, exp_data_path=None, exp_data=None):
        """
        绘制模拟结果与实验数据的对比。
        """
        if self.results is None:
            print("No simulation results found. Run scan first.")
            return

        exp_3x3 = None
        if exp_data is not None:
            exp_3x3 = exp_data
        elif exp_data_path is not None and os.path.exists(exp_data_path):
            exp_3x3 = np.load(exp_data_path)
        
        plt.figure(figsize=(10, 8))
        grid_h, grid_w = self.results.shape[0], self.results.shape[1]
        
        plot_idx = 1
        for i in range(min(3, grid_h)):
            for j in range(min(3, grid_w)):
                plt.subplot(3, 3, plot_idx)
                plt.plot(self.results[i, j, :], label='PMCX')
                
                if exp_3x3 is not None and exp_3x3.shape == self.results.shape:
                     plt.plot(exp_3x3[i, j, :], label='Exp')
                
                plt.ylim([0, 1.1])
                plt.grid(True)
                if plot_idx == 1:
                    plt.legend()
                plot_idx += 1
        
        mua = self.cfg['prop'][1][0] if self.cfg else 0
        mus = self.cfg['prop'][1][1] if self.cfg else 0
        plt.suptitle(f"PMCX vs Exp (mua={mua:.4f}, mus={mus:.2f}), Foam Size={self.foam_size}mm, Res={self.mcx_unitinmm}mm")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 测试代码
    base_folder = 'F:/OneDrive/UK_projects_local/project in UK 2024/diffuse_experiment241021/zhiguan_data/XL_measurement_061224'
    obj_file = 'X_061224.bmp'
    sens_file = 'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy'
    irf_file = 'IRF_gain0.7_timebin15ps_2000bins_061224.npy'

    # 1. 实例化
    sim = MCXFoamSimulator(base_folder, unit_mm=0.5, foam_size_mm=126)
    
    try:
        sim.load_and_process_resources(obj_file, sens_file, irf_file)
        sim.build_volume(thickness_mm=50)
        
        # 2. 运行第一次模拟
        print("--- First Run ---")
        sim.run_grid_scan(mu_a=0.0019, mu_s=1.48, grid_shape=(3,3))
        
        # 3. 手动释放（可选，因为循环内部已经有释放了，但如果想彻底清空volume可以用这个）
        # sim.release_memory() 
        # sim.build_volume(thickness_mm=50) # 如果释放了需要重新build
        
        # 4. 运行第二次模拟 (测试连续运行是否崩溃)
        print("--- Second Run ---")
        sim.run_grid_scan(mu_a=0.0025, mu_s=1.50, grid_shape=(3,3))
        
    except Exception as e:
        print(f"Main loop crashed: {e}")
    finally:
        # 最后清理
        sim.release_memory()
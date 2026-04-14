# -*- coding: utf-8 -*-
# 这个脚本实现了一个基于 MCX 的光学属性拟合工具，专门针对 10cm 厚度的样品。
# 主要功能包括：
# 1. 加载系统固有资源（IRF 和 Sensitivity Map）
# 2. 设置目标类型（如 X, L, I, G, H, T）并加载对应的实验数据
# 3. 定义前向模型，使用 MCX 进行仿真，并将结果与实验数据进行拟合
# 4. 计算拟合的 RMSE，并在结果图中显示
# author: zhiguan 2026-04-14

import numpy as np
import matplotlib.pyplot as plt
import pmcx
from scipy.optimize import curve_fit
from cv2 import resize
from scipy.signal import fftconvolve
import cv2
import os
import gc

class MCXPropertyFitter:
    def __init__(self, unit_mm=1.0, thickness_mm=100):
        """
        初始化拟合器。
        """
        self.unit_mm = unit_mm
        self.thickness_mm = thickness_mm
        
        # 内部容器
        self.irf_down = None
        self.sensitivity_pad = None
        self.exp_data_processed = None
        self.fit_weights = None
        self.vol = None
        self.cfg = {}
        
        # 结果容器
        self.fitted_params = None 
        self.best_fit_curves = None
        self.rmse = None  # <--- [新增] 存储 RMSE

        # [配置 1] 系统固有文件
        self.data_root = r'F:/OneDrive/foam_imaging_project/10cm_data'
        self.image_root = os.path.join(self.data_root, 'images')
        self.system_files = {
            'irf': os.path.join(self.data_root, 'IRF_gain0.7_timebin15ps_2000bins_080426.npy'),
            'sens': os.path.join(self.data_root, 'sensitivity_map_gain0.7_85x85mm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg__080426.npy')
        }

        # [配置 2] 目标类型配置
        self.target_config = {
            'None': {
                'obj_file': None, 
                'exp_file': os.path.join(self.data_root, 'noabs_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy')
            },
            'X': {
                'obj_file': os.path.join(self.image_root, 'X_exp2685ms.bmp'),
                'exp_file': os.path.join(self.data_root, 'X_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy') 
            },
            'L': {
                'obj_file': os.path.join(self.image_root, 'L_exp2685ms.bmp'),
                'exp_file': os.path.join(self.data_root, 'L_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy') 
            },
            'I':{
                'obj_file': os.path.join(self.image_root, 'i_exp2685ms.bmp'),
                'exp_file': os.path.join(self.data_root, 'i_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy')

            },
            'G':{
                'obj_file': os.path.join(self.image_root, 'G_exp2685ms.bmp'),
                'exp_file': os.path.join(self.data_root, 'G_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy')

            },
            'H':{
                'obj_file': os.path.join(self.image_root, 'H_exp2685ms.bmp'),
                'exp_file': os.path.join(self.data_root, 'H_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy')

            },
            'T':{
                'obj_file': os.path.join(self.image_root, 'T_exp2685ms.bmp'),
                'exp_file': os.path.join(self.data_root, 'T_gain0.7_5x5cm_3x3points_pol110deg_expo6.0sec_binWidth15ps_binNum2000_080426.npy')

            },

        }

    def _hist_downsample(self, hist_15ps, target_length=121, dt_old=15.0, dt_new=100.0):
        """
        将 15ps 数据重采样到 100ps，并强制输出固定长度。
        根据 target_length 自动计算所需的时间范围，并从原始数据中截取。
        
        参数:
            hist_15ps (array): 原始高分辨数据 (1D)
            target_length (int): 期望输出的数组长度 (例如 121)
            dt_old (float): 原始时间分辨率 (ps)，默认 15
            dt_new (float): 目标时间分辨率 (ps)，默认 100
            
        返回:
            hist_new (array): 长度严格等于 target_length 的数组
            time_new (array): 对应的时间轴
        """
        
        # 1. 计算所需的总时间长度 (Total Duration needed)
        required_duration = target_length * dt_new
        
        # 2. 检查原始数据够不够长
        total_input_time = len(hist_15ps) * dt_old
        if total_input_time < required_duration:
            raise ValueError(f"原始数据长度不足！\n"
                            f"需要时间: {required_duration} ps\n"
                            f"原始数据仅有: {total_input_time} ps ({len(hist_15ps)} points)")

        # 3. 构建原始数据的累积分布 (Cumulative Sum)
        # 这是一个物理上连续的函数：F(t) = 直到时间 t 的总光子数
        old_edges = np.arange(len(hist_15ps) + 1) * dt_old
        cumulative_counts = np.r_[0, np.cumsum(hist_15ps)]
        
        # 4. 生成目标数据的“时间边界” (Edges)
        # 这里我们严格生成 target_length + 1 个边界，以保证 diff 之后剩下 target_length 个点
        new_edges = np.arange(target_length + 1) * dt_new
        
        # 5. 【核心】在原始累积曲线上进行插值
        # np.interp 会自动根据 new_edges 的范围，只取原始数据中对应的那一段
        # 原始数据后面多余的部分（超过 required_duration 的部分）会被自动忽略
        new_cumulative = np.interp(new_edges, old_edges, cumulative_counts)
        
        # 6. 微分还原回直方图 (Diff)
        # 长度变为 target_length
        hist_new = np.diff(new_cumulative)
        # hist_new = hist_new/hist_new.max()  # 这行代码注释 不能在此处归一化
        # 7. 生成对应的时间轴 (取中心点)
        # time_new = new_edges[:-1] + (dt_new / 2)
        
        return hist_new

    def initialize_system(self):
        """加载系统通用资源"""
        print("Initializing system resources...")
        
        # Load IRF
        irf_name = self.system_files['irf']
        irf_path = irf_name
        if not os.path.exists(irf_path): raise FileNotFoundError(f"IRF not found: {irf_path}")  
        irf_raw = np.load(irf_path)[:850]
        irf_down = self._hist_downsample(irf_raw)
        irf_down = irf_down / irf_down.max()  # normalize here 
        self.irf_down = irf_down

        # plot the IRF to check
        plt.figure()
        plt.plot(self.irf_down)
        plt.show()

        # Load Sensitivity
        sens_name = self.system_files['sens']
        sens_path = sens_name
        if not os.path.exists(sens_path): raise FileNotFoundError(f"Sens map not found: {sens_path}")
        sens_raw = np.load(sens_path)
        if sens_raw.ndim == 3: sens_raw = sens_raw.sum(2)
        sens_raw = sens_raw / sens_raw.max()
        sens_raw = np.fliplr(sens_raw)
        sens_resized = resize(sens_raw, (85, 85), interpolation=cv2.INTER_LINEAR)
        self.sensitivity_pad = np.zeros((250, 250))
        self.sensitivity_pad[125-42:125+43, 125-42:125+43] = sens_resized
        print("System resources (IRF & Sens) ready.")

        plt.figure()
        plt.imshow(self.sensitivity_pad, cmap='hot')
        plt.show()

    def set_target(self, target_type='None', visualize=False):
        """设置目标类型并加载数据"""
        target_str = str(target_type)
        if target_str not in self.target_config: raise ValueError(f"Unknown target: {target_str}")
        config = self.target_config[target_str]
        print(f"--- Setting up target: {target_str} ---")

        # 1. Build Volume
        t_vox = int(self.thickness_mm / self.unit_mm)
        self.vol = np.ones([250, 250, t_vox + 1], dtype='uint8')
        self.vol[:, :, -1] = 0 
        
        if config['obj_file'] is not None:
            obj_path = config['obj_file']
            if os.path.exists(obj_path):
                print(f"Loading object: {config['obj_file']}")
                obj_raw = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)

                # cut obj figure here
                y_start = 271  # 起始 Y 坐标 (从上往下)
                y_end   = 696  # 结束 Y 坐标
                x_start = 774  # 起始 X 坐标 (从左往右)
                x_end   = 1198  # 结束 X 坐标

                obj_img_10x10 = obj_raw[y_start:y_end, x_start:x_end]

                obj_img_10x10 = obj_img_10x10/obj_img_10x10.max()
                bin_threshold = 0.3
                obj_img_10x10[obj_img_10x10<bin_threshold] = 0   # keep the obj as zero, a photon is terminated when moving from a non-zero to zero voxel.
                obj_img_10x10[obj_img_10x10>=bin_threshold] = 1

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
                
                # resize to 50x50
                obj_img_5x5_50x50 = cv2.resize(obj_img_5x5, (50, 50), interpolation=cv2.INTER_NEAREST_EXACT)

                obj_padded = cv2.copyMakeBorder(np.fliplr(obj_img_5x5_50x50), 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=1).astype('uint8')
                slice_idx = int(np.clip(np.round(t_vox / 2 - 1), 0, self.vol.shape[2] - 1))
                self.vol[:, :, slice_idx] = np.rot90(obj_padded, k=-1, axes=(0, 1))

                if visualize:
                    plt.figure()
                    plt.imshow(obj_img_5x5_50x50, cmap='gray')
                    plt.title(f"Loaded Object: {target_str}")
                    plt.axis('off')

                    plt.figure()
                    plt.imshow(obj_img_10x10,cmap='gray')
                    plt.show()

            else:
                print(f"Warning: Object file {obj_path} not found. Using homogeneous volume.")

        self.cfg = {
            'nphoton': 1e7, 'vol': self.vol, 'tstart': 0, 'tend': 12.1e-9, 'tstep': 0.1e-9,
            'srcdir': [0, 0, 1], 'unitinmm': self.unit_mm, 'issrcfrom0': 1, 'issaveref': 1,
            'isreflect': 0, 'detpos': [125, 125, t_vox, 4] 
        }

        # 2. Load Exp Data
        exp_path = config['exp_file']
        print(f"Loading experiment data: {config['exp_file']}")
        full_data = np.load(exp_path)
        
        # Spatial 31x31 -> 3x3
        if full_data.shape[0] == 31:
            indices = [0, 15, 30]
            exp_3x3 = full_data[np.ix_(indices, indices)] # 结果形状 (3, 3, 2000)
            self.exp_data_processed = np.zeros((3, 3, 121))
            for i in range(3):
                for j in range(3):
                    trace = exp_3x3[i, j, :1000] if exp_3x3.shape[2] >= 1000 else exp_3x3[i, j, :]
                    self.exp_data_processed[i, j] = self._hist_downsample(trace)
            self.exp_data_processed = self.exp_data_processed / self.exp_data_processed.max()
        elif full_data.shape[0] == 3 and full_data.shape[2]>121:
                self.exp_data_processed = np.zeros((3, 3, 121))
                for i in range(3):
                    for j in range(3):
                        trace = full_data[i, j, :1000] if full_data.shape[2] >= 1000 else full_data[i, j, :]
                        self.exp_data_processed[i, j] = self._hist_downsample(trace)
                self.exp_data_processed = self.exp_data_processed / self.exp_data_processed.max()
        else:
            self.exp_data_processed = full_data
        # Temporal 15ps -> 100ps

        self.fit_weights = self.exp_data_processed.max(axis=2)
        print("Target setup complete.")

    def _forward_model(self, dummy_x, mu_a, mu_s, RI = 1.5): # fix the RI to be 1
        """前向模型"""
        sim_3x3 = np.zeros((3, 3, 121))
        positions = [150, 125, 100] 
        self.cfg['prop'] = [[0, 0, 1, 1], [mu_a, mu_s, 0, RI]]
        
        try:
            for i, pos_y in enumerate(positions):
                for j, pos_x in enumerate(positions):
                    self.cfg['srcpos'] = [pos_x, pos_y, 0]
                    res = pmcx.run(self.cfg)
                    dref_bd = res['dref'][:, :, int(self.vol.shape[2]-1), :]
                    dref_bd_rot = np.rot90(dref_bd, k=1, axes=(0, 1))
                    curve = (dref_bd_rot * self.sensitivity_pad[:, :, None]).sum((0, 1))
                    limit = min(len(curve), 121)
                    sim_3x3[i, j, :limit] = curve[:limit]
                    del res, dref_bd, dref_bd_rot
            gc.collect()
            
            # show intermediate fitting result
            print(f"Simulated with mu_a={mu_a:.6f}, mu_s={mu_s:.4f}, RI={RI:.4f}")

            sim_3x3 = np.apply_along_axis(lambda m: fftconvolve(m, self.irf_down, mode='full'), axis=2, arr=sim_3x3)[:, :, :121]
            if sim_3x3.max() > 0: sim_3x3 /= sim_3x3.max()
            return (sim_3x3 / self.fit_weights[:, :, None]).flatten()
        except Exception as e:
            print(f"Sim Error: {e}")
            return np.zeros(3*3*121)

    def run_fitting(self, p0=[0.0019, 1.48, 1.44], method='lm', bounds=(-np.inf, np.inf)):
        """
        运行拟合。
        """
        if self.exp_data_processed is None: raise ValueError("Target not set.")
        print(f"Starting optimization (Method: {method})...")
        
        y_target = (self.exp_data_processed / self.fit_weights[:, :, None]).flatten()
        dummy_x = np.zeros_like(y_target)
        
        try:
            popt, pcov = curve_fit(self._forward_model, dummy_x, y_target, p0=p0, method=method, bounds=bounds, ftol=1e-4, xtol=1e-4)
            self.fitted_params = popt

            # 定义参数名和对应的格式
            param_names = ['mua', 'mus', 'RI']
            param_fmts  = ['.6f', '.4f', '.4f']
            # 自动配对并生成字符串
            # zip 会自动以最短的列表（即 popt 的长度）为准截断，不会报错
            parts = [f"{name}={val:{fmt}}" for name, fmt, val in zip(param_names, param_fmts, popt)]
            print(f"Fitted: {', '.join(parts)}")  

            # 生成最佳拟合曲线
            fitted_flat = self._forward_model(dummy_x, *popt)
            self.best_fit_curves = fitted_flat.reshape(3, 3, 121) * self.fit_weights[:, :, None]
            
            # === [新增] 计算 RMSE ===
            # RMSE = sqrt(mean((y_sim - y_exp)^2))
            mse = np.mean((self.exp_data_processed - self.best_fit_curves) ** 2)
            self.rmse = np.sqrt(mse)
            print(f"Final RMSE: {self.rmse:.6f}")
            # =======================

            return popt
        except Exception as e:
            print(f"Fitting Failed: {e}")
            return None
        
    def run_MCX(self, mu_a=0.0019, mu_s=1.48, RI=1.44):
        """
        运行单次 MCX 仿真 (不进行拟合)，直接使用给定参数。
        """
        print(f"Running single simulation with: mua={mu_a:.6f}, mus={mu_s:.4f}, RI={RI:.4f}")
        
        try:
            # 1. 准备参数
            params = [mu_a, mu_s, RI]
            
            # 2. 调用前向模型
            # 注意：_forward_model 通常设计为 (x, p1, p2, p3)，
            # 这里第一个参数传 None (假设你的模型内部不依赖这个 dummy x，或者你可以传一个占位符)
            sim_result_flat = self._forward_model(None, *params)
            
            # 3. 还原数据形状
            # 原代码逻辑中：拟合用的数据是被 flatten 且除以过权重的
            # 所以这里需要 reshape 回 (3, 3, 121) 并乘回权重，以得到真实的物理量曲线
            # 这里的 shape (3, 3, 121) 是根据你原代码推断的，请根据实际情况调整
            self.best_fit_curves = sim_result_flat.reshape(3, 3, 121) * self.fit_weights[:, :, None]
            
            # 保存当前使用的参数，以便后续绘图或记录使用
            self.fitted_params = np.array(params)

            # 4. 计算 RMSE (如果有实验数据作为参考)
            if self.exp_data_processed is not None:
                # RMSE = sqrt(mean((y_sim - y_exp)^2))
                mse = np.mean((self.exp_data_processed - self.best_fit_curves) ** 2)
                self.rmse = np.sqrt(mse)
                print(f"Simulation Done. RMSE against exp data: {self.rmse:.6f}")
            else:
                self.rmse = None
                print("Simulation Done. (No target data set, skipped RMSE calculation)")

            return self.best_fit_curves

        except Exception as e:
            print(f"Simulation Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_results(self):
        """
        绘制拟合结果对比图，标题包含 RMSE。
        """
        if self.best_fit_curves is None: return

        plt.figure(figsize=(12, 10))
        y_exp = self.exp_data_processed
        y_fit = self.best_fit_curves
        p = self.fitted_params
        
        # 格式化 RMSE 字符串
        rmse_str = f"{self.rmse:.6f}" if self.rmse is not None else "N/A"

        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, 3*i + j + 1)
                plt.plot(y_exp[i, j, :], 'r-', label='Exp')
                plt.plot(y_fit[i, j, :], 'b--', label='Fitted Sim')
                plt.ylim([0, 1.1])
                plt.grid(True, alpha=0.3)
                if i==0 and j==0: plt.legend()
        
        # 标题增加 RMSE 显示
        plt.suptitle(
            f'Fit Result (RMSE={rmse_str}):\n'
            f'mu_a={p[0]:.6f} mm$^{{-1}}$, mu_s={p[1]:.4f} mm$^{{-1}}$, RI={p[2]:.4f}', 
            fontsize=16
        )
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 使用示例
    
    fitter = MCXPropertyFitter()
    fitter.initialize_system()
    
    # 设置 X 物体
    fitter.set_target('T', visualize=True)
    
    # 运行拟合
    # fitter.run_fitting(
    #     p0=[0.0019, 1.48, 1.44], 
    #     method='trf', 
    #     bounds=([0, 0.1, 1], [0.1, 5, 2])
    # )
    fitter.run_MCX()
    # # 绘图（此时标题会显示 RMSE）
    fitter.plot_results() 

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
    def __init__(self, data_fold, unit_mm=1.0, thickness_mm=50):
        """
        初始化拟合器。
        """
        self.data_fold = data_fold
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
        self.system_files = {
            'irf': 'IRF_gain0.7_timebin15ps_2000bins_061224.npy',
            'sens': 'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy'
        }

        # [配置 2] 目标类型配置
        self.target_config = {
            'None': {
                'obj_file': None, 
                'exp_file': 'noabs1_gain0.7_5x5cm_31x31points_pol50deg_expo1.0sec_binWidth15ps_binNum2000_061224.npy'
            },
            'X': {
                'obj_file': 'obj_x_5x5cm_50x50_PMT_view.npy',
                'exp_file': 'exp_real_X_3x3_data.npy' 
            },
            'L': {
                'obj_file': 'obj_l_5x5cm_50x50_PMT_view.npy',
                'exp_file': 'exp_real_L_3x3_data.npy' 
            }
        }

    def _downsample_1d(self, data_array, target_len=121, orig_bin=15, target_bin=100):
        """辅助函数：重采样"""
        needed_len = np.round(target_len * (target_bin / orig_bin)).astype('int')
        if len(data_array) < needed_len:
            needed_len = len(data_array)
        res = resize(data_array[:needed_len], (1, target_len), interpolation=cv2.INTER_NEAREST_EXACT)
        return np.squeeze(res)

    def initialize_system(self):
        """加载系统通用资源"""
        print("Initializing system resources...")
        
        # Load IRF
        irf_name = self.system_files['irf']
        irf_path = os.path.join(self.data_fold, irf_name)
        if not os.path.exists(irf_path): raise FileNotFoundError(f"IRF not found: {irf_path}")  
        irf_raw = np.load(irf_path)[:1000]
        irf_raw = irf_raw - irf_raw[:100].mean()
        irf_raw = irf_raw / irf_raw.max()
        self.irf_down = self._downsample_1d(irf_raw)

        # Load Sensitivity
        sens_name = self.system_files['sens']
        sens_path = os.path.join(self.data_fold, sens_name)
        if not os.path.exists(sens_path): raise FileNotFoundError(f"Sens map not found: {sens_path}")
        sens_raw = np.load(sens_path)
        if sens_raw.ndim == 3: sens_raw = sens_raw.sum(2)
        sens_raw = sens_raw / sens_raw.max()
        sens_raw = np.fliplr(sens_raw)
        sens_resized = resize(sens_raw, (80, 80), interpolation=cv2.INTER_LINEAR)
        self.sensitivity_pad = np.zeros((250, 250))
        self.sensitivity_pad[125-40:125+40, 125-40:125+40] = sens_resized
        print("System resources (IRF & Sens) ready.")

    def set_target(self, target_type='None'):
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
            obj_path = os.path.join(self.data_fold, config['obj_file'])
            if os.path.exists(obj_path):
                print(f"Loading object: {config['obj_file']}")
                obj_raw = np.load(obj_path)
                obj_padded = cv2.copyMakeBorder(obj_raw, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=1).astype('uint8')
                self.vol[:, :, 24] = np.rot90(obj_padded, k=-1, axes=(0, 1))
            else:
                print(f"Warning: Object file {obj_path} not found. Using homogeneous volume.")

        self.cfg = {
            'nphoton': 1e7, 'vol': self.vol, 'tstart': 0, 'tend': 12.1e-9, 'tstep': 0.1e-9,
            'srcdir': [0, 0, 1], 'unitinmm': self.unit_mm, 'issrcfrom0': 1, 'issaveref': 1,
            'detpos': [125, 125, t_vox, 4] 
        }

        # 2. Load Exp Data
        exp_path = os.path.join(self.data_fold, config['exp_file'])
        print(f"Loading experiment data: {config['exp_file']}")
        full_data = np.load(exp_path)
        
        if target_str == 'None':
            # Spatial 31x31 -> 3x3
            if full_data.shape[0] > 3:
                X_idx, Y_idx = np.meshgrid([0, 15, 30], [0, 15, 30])
                exp_3x3 = full_data[X_idx, Y_idx]
            else:
                exp_3x3 = full_data
            # Temporal 15ps -> 100ps
            self.exp_data_processed = np.zeros((3, 3, 121))
            for i in range(3):
                for j in range(3):
                    trace = exp_3x3[i, j, :1000] if exp_3x3.shape[2] >= 1000 else exp_3x3[i, j, :]
                    self.exp_data_processed[i, j] = self._downsample_1d(trace)
        else:
            self.exp_data_processed = full_data

        self.exp_data_processed = self.exp_data_processed / self.exp_data_processed.max()
        self.fit_weights = self.exp_data_processed.max(axis=2)
        print("Target setup complete.")

    def _forward_model(self, dummy_x, mu_a, mu_s, RI):
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
            print(f"Fitted: mua={popt[0]:.6f}, mus={popt[1]:.4f}, RI={popt[2]:.4f}")
            
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
                plt.plot(y_exp[i, j, :], 'b-', label='Exp')
                plt.plot(y_fit[i, j, :], 'r--', label='Fitted Sim')
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
    data_dir = r'F:/OneDrive/foam_imaging_project/pmcx_foam/experiment_data/'
    
    fitter = MCXPropertyFitter(data_dir)
    fitter.initialize_system()
    
    # 设置 X 物体
    fitter.set_target('X')
    
    # 运行拟合
    fitter.run_fitting(
        p0=[0.0019, 1.48, 1.44], 
        method='trf', 
        bounds=([0, 0.1, 1], [0.1, 5, 2])
    )
    
    # 绘图（此时标题会显示 RMSE）
    fitter.plot_results()
import numpy as np
import matplotlib.pyplot as plt
import pmcx
from cv2 import resize
import cv2
import os
from tqdm import tqdm
from scipy.signal import fftconvolve
import gc

class MCXTargetSimulator:
    def __init__(self, data_fold, unit_mm=1.0):
        """
        初始化仿真器。
        
        Args:
            data_fold (str): 数据所在的根目录。
            unit_mm (float): MCX 的体素大小 (mm)。代码逻辑默认基于 1mm/vox (250x250 volume)。
        """
        self.data_fold = data_fold
        self.unit_mm = unit_mm
        
        # 内部变量
        self.sensitivity_pad = None
        self.irf_down = None
        self.vol = None
        self.exp_data = None # 存储当前目标的实验真值
        self.target_name = None # 'X' or 'L'
        self.cfg = {}
        self.sim_results = None

        # 文件路径映射 (你需要根据实际文件名修改这里)
        self.file_mapping = {
            'X': {
                'obj_file': 'obj_x_5x5cm_50x50_PMT_view.npy',
                'exp_file': 'exp_real_X_3x3_data.npy' # 假设真值文件名
            },
            'L': {
                'obj_file': 'obj_l_5x5cm_50x50_PMT_view.npy', # 假设L的文件名
                'exp_file': 'exp_real_L_3x3_data.npy' # 假设L的真值文件名
            }
        }

    def load_system_resources(self, sens_filename = 'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy', 
                              irf_filename = 'IRF_gain0.7_timebin15ps_2000bins_061224.npy'):
        """
        加载通用的系统资源：灵敏度图 (Sensitivity Map) 和 仪器响应函数 (IRF)。
        这两个文件通常与具体物体形状无关。
        """
        # 1. 处理 Sensitivity Map
        sens_path = os.path.join(self.data_fold, sens_filename)
        if sens_filename.endswith('.npy'):
            sensitivity = np.load(sens_path)
            if sensitivity.ndim == 3:
                sensitivity = sensitivity.sum(2)
        else:
            raise ValueError("Sensitivity map needs to be .npy")

        sensitivity = sensitivity / sensitivity.max()
        sensitivity = np.fliplr(sensitivity) # PMT view flip
        
        # Resize to 80x80 (8cm FOV)
        sensitivity = resize(sensitivity, (80, 80), interpolation=cv2.INTER_LINEAR)
        
        # Padding to 250x250 (Place in center)
        # 125 is center, 80/2 = 40. Range: 85 to 165
        self.sensitivity_pad = np.zeros((250, 250))
        center_idx = 125
        half_width = 40
        self.sensitivity_pad[center_idx-half_width:center_idx+half_width, 
                             center_idx-half_width:center_idx+half_width] = sensitivity

        # 2. 处理 IRF
        irf_path = os.path.join(self.data_fold, irf_filename)
        IRF = np.load(irf_path)[:1000]
        # Downsample logic: 15ps -> 100ps
        target_len = 121
        samples_needed = np.round(target_len * 100 / 15).astype('int')
        IRF_down = resize(IRF[:samples_needed], (1, target_len), interpolation=cv2.INTER_NEAREST_EXACT)
        IRF_down = np.squeeze(IRF_down)
        self.irf_down = IRF_down / IRF_down.max()

        print("System resources (Sensitivity Map & IRF) loaded.")

    def set_target_object(self, target_type, custom_obj_path=None, custom_exp_path=None):
        """
        切换目标物体（X 或 L），并构建对应的 MCX Volume。
        
        Args:
            target_type (str): 'X' 或 'L'。
            custom_obj_path (str, optional): 如果不在默认映射中，可手动传入物体npy路径。
            custom_exp_path (str, optional): 手动传入实验真值路径。
        """
        self.target_name = target_type
        
        # 1. 确定文件路径
        if target_type in self.file_mapping:
            obj_fname = self.file_mapping[target_type]['obj_file']
            exp_fname = self.file_mapping[target_type]['exp_file']
            obj_path = os.path.join(self.data_fold, obj_fname)
            exp_path = os.path.join(self.data_fold, exp_fname)
        else:
            if custom_obj_path is None:
                raise ValueError(f"Unknown target type '{target_type}' and no custom path provided.")
            obj_path = custom_obj_path
            exp_path = custom_exp_path

        # 2. 加载物体形状
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Object file not found: {obj_path}")
        
        print(f"Loading object: {target_type} from {os.path.basename(obj_path)}")
        obj_in_mcx_PMT = np.load(obj_path)
        
        # Padding (50x50 -> 250x250)
        # Pad 100 on each side
        obj_padded = cv2.copyMakeBorder(
            obj_in_mcx_PMT, 
            100, 100, 100, 100, 
            cv2.BORDER_CONSTANT, 
            value=1
        ).astype('uint8')

        # 3. 构建 MCX Volume
        thickness = 50 # mm
        t_vox = int(thickness / self.unit_mm)
        
        # 显式释放旧内存
        if self.vol is not None:
            del self.vol
            gc.collect()
            
        self.vol = np.ones([250, 250, t_vox + 1], dtype='uint8')
        self.vol[:, :, -1] = 0
        
        # 插入物体 (Z=24 based on your code)
        # 旋转物体以匹配 MCX 坐标系
        self.vol[:, :, 24] = np.rot90(obj_padded, k=-1, axes=(0, 1))

        # 4. 加载对应的实验真值 (用于后续绘图对比)
        if exp_path and os.path.exists(exp_path):
            self.exp_data = np.load(exp_path)
            print(f"Experimental data loaded for {target_type}.")
        else:
            print(f"Warning: Experimental data not found at {exp_path}")
            self.exp_data = None

    def run_simulation(self, mu_a, mu_s, RI, save_result=False):
        """
        使用指定的 mu_a 和 mu_s 运行仿真。
        """
        if self.vol is None:
            raise ValueError("Volume not built. Call set_target_object first.")

        # 配置 MCX
        self.cfg = {
            'nphoton': 1e7,
            'vol': self.vol,
            'tstart': 0,
            'tend': 12.1e-9,
            'tstep': 0.1e-9,
            'srcdir': [0, 0, 1],
            'unitinmm': self.unit_mm,
            'issrcfrom0': 1,
            'issaveref': 1,
            'prop': [
                [0, 0, 1, 1],          # background
                [mu_a, mu_s, 0, RI]  # volume
            ]
        }

        # 扫描位置 (中心 125, 范围 +/- 25)
        pmcx_result = np.zeros((3, 3, 121))
        src_pos_range = np.linspace(125+25, 125-25, 3) 

        print(f"Running simulation for Target={self.target_name}, mua={mu_a}, mus={mu_s}...")
        
        try:
            for i, pos_y in tqdm(enumerate(src_pos_range), total=3):
                for j, pos_x in enumerate(src_pos_range):
                    self.cfg['srcpos'] = [pos_x, pos_y, 0]
                    
                    res = pmcx.run(self.cfg)
                    
                    # 提取边界并处理
                    dref_bd = res['dref'][:,:,int(self.vol.shape[2]-1),:]
                    dref_bd_rot = np.rot90(dref_bd, k=1, axes=(0,1))
                    
                    # 应用 Sensitivity Map
                    curve = (dref_bd_rot * self.sensitivity_pad[:,:,None]).sum((0,1))
                    
                    # 截取时间窗口
                    limit = min(len(curve), 121)
                    pmcx_result[i, j, :limit] = curve[:limit]

                    # 内存清理
                    del res
                    del dref_bd
                    del dref_bd_rot
            
            gc.collect()

            # 卷积 IRF
            pmcx_result = np.apply_along_axis(
                lambda m: fftconvolve(m, self.irf_down, mode='full'), 
                axis=2, 
                arr=pmcx_result
            )[:, :, :len(self.irf_down)]

            # 归一化
            if pmcx_result.max() > 0:
                pmcx_result = pmcx_result / pmcx_result.max()
            
            self.sim_results = pmcx_result
            
            if save_result:
                save_name = f'sim_MCX_obj_{self.target_name}_mua{mu_a}_mus{mu_s}.npy'
                save_path = os.path.join(self.data_fold, save_name)
                np.save(save_path, pmcx_result)
                print(f"Result saved to {save_name}")
                
            return pmcx_result
            
        except Exception as e:
            print(f"Simulation crashed: {e}")
            gc.collect()
            return None

    # [修改点 2] 增加 RMSE 计算
    def calculate_rmse(self):
        """内部辅助函数：计算当前仿真结果与实验真值的RMSE"""
        if self.sim_results is None or self.exp_data is None:
            return None
        
        # 提取实验数据中对应的 3x3 点
        # 假设实验数据较大(如31x31)，需要提取 [0, 15, 30] 索引
        # 如果实验数据本身就是 3x3，则直接使用
        exp_subset = np.zeros_like(self.sim_results)
        
        if self.exp_data.shape[0] > 3:
            indices = [0, 15, 30]
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    # 确保不越界
                    r_idx = min(idx_i, self.exp_data.shape[0]-1)
                    c_idx = min(idx_j, self.exp_data.shape[1]-1)
                    exp_subset[i, j, :] = self.exp_data[r_idx, c_idx, :]
        else:
            exp_subset = self.exp_data

        # 再次确保归一化（对比形状）
        if exp_subset.max() > 0: exp_subset /= exp_subset.max()
        if self.sim_results.max() > 0: self.sim_results /= self.sim_results.max()

        # 计算 RMSE (Root Mean Square Error)
        # 这里的 RMSE 是基于所有时间点和所有空间点的平均
        mse = np.mean((self.sim_results - exp_subset) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def plot_comparison(self, show_plot=True):
        """
        绘制对比图并返回 RMSE
        Args:
            show_plot: 是否显示图像，批量跑循环时可以设为 False
        Returns:
            rmse: 计算出的误差值
        """
        rmse = self.calculate_rmse()
        rmse_str = f"{rmse:.6f}" if rmse is not None else "N/A"
        
        if not show_plot:
            return rmse

        plt.figure(figsize=(10, 8))
        indices = [0, 15, 30]
        
        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, 3*i + j + 1)
                plt.plot(self.sim_results[i, j, :], 'r-', label='Sim' if i==0 and j==0 else "")
                
                if self.exp_data is not None:
                    # 绘图逻辑同 calculate_rmse
                    if self.exp_data.shape[0] > 3:
                        plt.plot(self.exp_data[indices[i], indices[j], :], 'b--', label='Exp' if i==0 and j==0 else "")
                    else:
                        plt.plot(self.exp_data[i, j, :], 'b--', label='Exp' if i==0 and j==0 else "")
                
                plt.ylim([0, 1.1])
                plt.grid(True, alpha=0.3)
                if i==0 and j==0: plt.legend()

        mua = self.cfg['prop'][1][0] if self.cfg else 0
        mus = self.cfg['prop'][1][1] if self.cfg else 0
        ri = self.cfg['prop'][1][3] if self.cfg else 0
        
        plt.suptitle(f"Target: {self.target_name} | mua={mua}, mus={mus}, RI={ri}\nRMSE: {rmse_str}", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return rmse
# %% 使用示例
if __name__ == "__main__":
    # 1. 设置路径
    data_dir = r'F:/OneDrive/foam_imaging_project/pmcx_foam/experiment_data/'
    # 注意：你需要确保文件夹里有 'obj_x...npy', 'obj_l...npy', 'sensitivity...npy', 'IRF...npy'
    
    sens_file = 'sensitivity_map_gain0.7_8x8cm_51x51points_binWidth15ps_expo0.1sec_binNum2000_40deg_061224.npy'
    irf_file = 'IRF_gain0.7_timebin15ps_2000bins_061224.npy'

    # 2. 实例化
    sim = MCXTargetSimulator(data_dir, unit_mm=1.0)
    
    # 3. 加载一次系统资源
    sim.load_system_resources(sens_file, irf_file)

    # ================= Case 1: 模拟 X 物体 =================
    # 假设你的X真值文件叫 exp_real_X_31x31.npy (对应你的代码索引逻辑)
    # 你可以在类初始化时修改 file_mapping，或者直接在这里确保文件名对上
    
    sim.set_target_object('X') 
    
    # 运行仿真 (调节 mu_a, mu_s)
    sim.run_simulation(mu_a=0.0019, mu_s=1.48)
    sim.plot_comparison()

    # ================= Case 2: 切换到 L 物体 =================
    # 自动加载 L 的形状和 L 的真值
    # sim.set_target_object('L') 
    
    # 运行仿真 (尝试不同的参数)
    # sim.run_simulation(mu_a=0.0025, mu_s=1.60)
    # sim.plot_comparison()
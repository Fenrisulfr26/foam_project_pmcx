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
                'exp_file': 'X_gain0.7_5x5cm_31x31points_pol50deg_expo0.5sec_binWidth15ps_binNum2000_061224.npy' # X真值文件名
            },
            'L': {
                'obj_file': 'obj_l_5x5cm_50x50_PMT_view.npy', # 假设L的文件名
                'exp_file': 'L_gain0.7_5x5cm_31x31points_pol50deg_expo0.5sec_binWidth15ps_binNum2000_061224.npy' #L的真值文件名
            },
            'None': {
                'obj_file': None,
                'exp_file': 'noabs1_gain0.7_5x5cm_31x31points_pol50deg_expo1.0sec_binWidth15ps_binNum2000_061224.npy'
            }
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
        
        # 7. 生成对应的时间轴 (取中心点)
        # time_new = new_edges[:-1] + (dt_new / 2)
        
        return hist_new    

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
        irf_raw = np.load(irf_path)[:1000]
        # Downsample logic: 15ps -> 100ps
        irf_raw = irf_raw - irf_raw[:100].mean() # git rid of dark count
        irf_down = self._hist_downsample(irf_raw)
        irf_down = irf_down / irf_down.max()  # normalize here
        self.irf_down = irf_down

        print("System resources (Sensitivity Map & IRF) loaded.")

            
    def set_target_object(self, target_type, custom_obj_path=None, custom_exp_path=None):
        """
        切换目标物体，构建 MCX Volume，并处理对应的实验数据。
        
        支持的 target_type:
        1. 'None': 均质体，自动从 31x31x2000 数据中提取 3x3 并下采样。
        2. 字符串 (如 'A', 'B'): 根据 file_mapping 查找对应的 obj 和 exp 文件。
        3. Numpy Array (图像): 直接使用该图像矩阵作为掩膜，跳过 obj 文件读取。
        """
        # --- 0. 判断输入类型 ---
        is_custom_image = isinstance(target_type, np.ndarray)
        
        if is_custom_image:
            self.target_name = 'Custom_Image'
        else:
            self.target_name = str(target_type)

        # --- 1. 确定文件路径 ---
        obj_path = None
        exp_path = None
        
        if not is_custom_image:
            # 如果是字符串，则去字典里找路径
            mapping = self.file_mapping.get(self.target_name, {})
            
            if target_type == 'None':
                if 'exp_file' in mapping:
                    exp_path = os.path.join(self.data_fold, mapping['exp_file'])
                elif custom_exp_path:
                    exp_path = custom_exp_path
            else:
                if 'obj_file' in mapping and 'exp_file' in mapping:
                    obj_path = os.path.join(self.data_fold, mapping['obj_file'])
                    exp_path = os.path.join(self.data_fold, mapping['exp_file'])
                else:
                    if custom_obj_path is None:
                        raise ValueError(f"Unknown target type '{target_type}' and no custom path provided.")
                    obj_path = custom_obj_path
                    exp_path = custom_exp_path
        else:
            # 如果直接传入图像，实验数据路径只能依赖手动传入的 custom_exp_path
            exp_path = custom_exp_path

        # --- 2. 构建基础 MCX Volume ---
        thickness = 50 # mm
        t_vox = int(thickness / self.unit_mm)
        
        if self.vol is not None:
            del self.vol
            gc.collect()
            
        self.vol = np.ones([250, 250, t_vox + 1], dtype='uint8')
        self.vol[:, :, -1] = 0 

        # 配置 MCX
        self.cfg = {
            'nphoton': 1e7,
            'vol': self.vol,
            'tstart': 0,
            'tend': 12.1e-9,
            'tstep': 0.1e-9,
            'srcdir':[0, 0, 1],
            'unitinmm': self.unit_mm,
            'isreflect': 0,
            'issrcfrom0': 1,
            'issaveref': 1,
            'prop':[
                [0, 0, 1, 1],          # background[mu_a, mu_s, 0, RI]    # volume
            ]
        }

        # --- 3. 插入物体 ---
        if is_custom_image:
            # 【新增逻辑】如果传入的是图片，直接使用
            print("Target is an image array: Using provided array as mask.")
            obj_in = target_type
            obj_padded = cv2.copyMakeBorder(obj_in, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=1).astype('uint8')
            self.vol[:, :, 24] = np.rot90(obj_padded, k=-1, axes=(0, 1))
            
        elif target_type != 'None':
            # 【原有逻辑】如果传入的是字符串，加载文件
            if not os.path.exists(obj_path):
                raise FileNotFoundError(f"Object file not found: {obj_path}")
            
            print(f"Loading object: {self.target_name} from {os.path.basename(obj_path)}")
            obj_in = np.load(obj_path)
            obj_padded = cv2.copyMakeBorder(obj_in, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=1).astype('uint8')
            self.vol[:, :, 24] = np.rot90(obj_padded, k=-1, axes=(0, 1))
            
        else:
            print("Target is None: Using homogeneous volume.")

        # --- 4. 加载并处理实验真值 (保持不变) ---
        if exp_path and os.path.exists(exp_path):
            print(f"Loading experiment data: {os.path.basename(exp_path)}")
            raw_data = np.load(exp_path) # 预期形状: (31, 31, 2000)
                        
            if raw_data.shape[0] > 3:
                print("Processing data: Extracting 3x3 spatial points and downsampling time...")
                indices =[0, 15, 30]
                
                if raw_data.shape[0] >= 31 and raw_data.shape[1] >= 31:
                    spatial_subset = raw_data[np.ix_(indices, indices)] 
                else:
                    print(f"Raw data shape {raw_data.shape} is small, assuming already spatial sliced.")
                    spatial_subset = raw_data

                self.exp_data = np.zeros((3, 3, 121))
                
                for r in range(3):
                    for c in range(3):
                        trace = spatial_subset[r, c, :]
                        trace_down = self._hist_downsample(trace)
                        self.exp_data[r, c, :] = np.squeeze(trace_down)

                self.exp_data = self.exp_data / self.exp_data.max()
                print(f"Data processed to shape: {self.exp_data.shape}")
                
            else:
                self.exp_data = raw_data
                print(f"Loaded pre-processed data shape: {self.exp_data.shape}")
                
        else:
            print(f"Warning: Experimental data not found or not provided.")
            self.exp_data = None


    def run_simulation(self, mu_a = 0.0019, mu_s = 1.48, RI = 1.44,  save_result=False, save_exp = False):
        """
        使用指定的 mu_a 和 mu_s 运行仿真。
        """
        # 更新所用的光学参数

        new_prop = np.array([mu_a, mu_s, 0, RI])
        self.cfg['prop'] = np.vstack((self.cfg['prop'], new_prop))
        
        print(f"Running simulation for Target={self.target_name}, mua={mu_a}, mus={mu_s}, RI = {RI}...")

        if self.vol is None:
            raise ValueError("Volume not built. Call set_target_object first.")


        # 扫描位置 (中心 125, 范围 +/- 25)
        pmcx_result = np.zeros((3, 3, 121))
        src_pos_range = np.linspace(125+25, 125-25, 3) 
        
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
                save_name = f"sim_MCX_obj_{self.target_name}_mua{self.cfg['prop'][1][0]}_mus{self.cfg['prop'][1][1]}.npy"
                save_path = os.path.join(self.data_fold, save_name)
                np.save(save_path, pmcx_result)
                print(f"Result saved to {save_name}")
                
             
            if save_result:
                save_name = f"expdata_obj_{self.target_name}.npy"
                save_path = os.path.join(self.data_fold, save_name)
                np.save(save_path, self.exp_data)
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
                if self.exp_data is not None:
                    # 绘图逻辑同 calculate_rmse
                    if self.exp_data.shape[0] > 3:
                        plt.plot(self.exp_data[indices[i], indices[j], :], 'r-', label='Exp' if i==0 and j==0 else "")
                    else:
                        plt.plot(self.exp_data[i, j, :], 'r-', label='Exp' if i==0 and j==0 else "")
                plt.plot(self.sim_results[i, j, :], 'b--', label='Sim' if i==0 and j==0 else "")
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
    # sim.run_simulation(mu_a=0.0019, mu_s=1.48)
    sim.plot_comparison()

    # ================= Case 2: 切换到 L 物体 =================
    # 自动加载 L 的形状和 L 的真值
    # sim.set_target_object('L') 
    
    # 运行仿真 (尝试不同的参数)
    # sim.run_simulation(mu_a=0.0025, mu_s=1.60)
    # sim.plot_comparison()
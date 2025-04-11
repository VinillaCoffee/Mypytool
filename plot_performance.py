import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
import seaborn as sns

# 设置风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # 更新版本的matplotlib使用此样式名称
except:
    plt.style.use('seaborn-whitegrid')  # 旧版本的matplotlib
sns.set_context("paper", font_scale=1.5)

# 方法名称映射
METHOD_NAMES = {
    'prune': 'Prune',
    'r_walker': 'RWalk',
    'lwf': 'LWF',
    'grow': 'Grow',
    'vcl': 'VCL',
    'lora': 'CoD',
    'packnet': 'PackNet',
    'agem': 'A-GEM',
    'mas': 'MAS',
    'ewc': 'EWC',
    'pm': 'PM',
    'l2': 'L2',
    'finetuning': 'Finetuning'
}

# 颜色映射，保持与图例一致
COLOR_MAP = {
    'PM': 'tab:red',
    'RWalk': 'tab:orange',
    'PackNet': 'tab:green',
    'Finetuning': 'tab:cyan',
    'L2': 'tab:blue',
    'MAS': 'tab:purple',
    'Prune': 'tab:pink',
    'LWF': 'tab:olive',
    'LoRA': 'tab:brown',
    'EWC': 'tab:gray',
    'A-GEM': 'mediumaquamarine',
    'VCL': 'slategray',
    'Grow': 'orchid'
}

def extract_success_rates(csv_path):
    """提取CSV文件中的成功率数据"""
    try:
        # 直接读取CSV文件内容
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        iterations = []
        success_rates = []
        
        # 获取标题行来确定哪些列是Success列
        header = lines[0].strip().split(',')
        success_indices = [i for i, col in enumerate(header) if col.endswith('Success')]
        
        # 解析每个Iteration行及其成功率
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('Iteration'):
                i += 1  # 跳过标题行
                continue
                
            # 检查是否是包含Iteration的行
            if line and ',' in line and not line.startswith('hammer') and not line.startswith('push') and not line.startswith('faucet') and not line.startswith('stick') and not line.startswith('handle') and not line.startswith('shelf') and not line.startswith('window') and not line.startswith('peg'):
                parts = line.split(',')
                # 确保有足够的列
                if len(parts) > 1:
                    try:
                        # 获取迭代次数
                        iteration = int(parts[0])
                        
                        # 只获取Success列的值 (根据标题确定)
                        success_values = []
                        for idx in success_indices:
                            if idx < len(parts) and parts[idx].strip():
                                try:
                                    # 直接使用原始success值
                                    success_values.append(float(parts[idx]))
                                except ValueError:
                                    # 忽略无法转换的值
                                    pass
                        
                        if success_values:
                            avg_success = np.mean(success_values)
                            iterations.append(iteration)
                            success_rates.append(avg_success)
                    except Exception as e:
                        print(f"处理行 '{line}' 时出错: {e}")
            i += 1
        
        if not iterations:
            print(f"在 {csv_path} 中没有找到可用的迭代数据")
            return None, None
            
        return np.array(iterations, dtype=np.float64), np.array(success_rates, dtype=np.float64)
    
    except Exception as e:
        print(f"处理 {csv_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def plot_performance_curves():
    """绘制所有方法的平均成功率曲线"""
    save_dir = 'save'
    # 获取所有子文件夹
    directories = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    
    plt.figure(figsize=(12, 8))
    
    # 存储每个方法的数据，用于计算多个种子的平均值和标准差
    methods_data = {}
    
    for directory in directories:
        # 提取方法名称
        match = re.search(r'cw10-([a-zA-Z_]+)-', directory)
        if not match:
            continue
            
        method_key = match.group(1)
        method_name = METHOD_NAMES.get(method_key, method_key)
        
        # 查找progress.csv文件
        csv_path = os.path.join(save_dir, directory, 'progress.csv')
        if not os.path.exists(csv_path):
            print(f"在 {directory} 中没有找到progress.csv文件")
            continue
        
        # 提取成功率数据
        iterations, success_rates = extract_success_rates(csv_path)
        if iterations is None or success_rates is None:
            continue
        
        print(f"提取 {method_name} 的数据: 迭代次数={len(iterations)}个, 成功率={len(success_rates)}个")
        
        # 存储数据
        if method_name not in methods_data:
            methods_data[method_name] = []
        
        # 存储(迭代次数, 成功率)对
        methods_data[method_name].append((iterations, success_rates))
    
    # 绘制每个方法的平均曲线和置信区间
    for method_name, data_list in methods_data.items():
        # 确保有数据
        if not data_list:
            print(f"方法 {method_name} 没有可用数据")
            continue
        
        # 获取最大迭代次数
        max_iter = max([iterations[-1] for iterations, _ in data_list])
        
        # 将所有种子的数据重新采样到相同的x轴刻度
        x = np.linspace(0, max_iter, 100)
        y_interp = []
        
        for iterations, success_rates in data_list:
            # 只考虑有效范围内的数据
            valid_indices = ~np.isnan(success_rates)
            if np.sum(valid_indices) < 2:  # 需要至少两个点进行插值
                print(f"方法 {method_name} 的数据点不足")
                continue
                
            # 进行插值
            try:
                y = np.interp(x, iterations[valid_indices], success_rates[valid_indices])
                y_interp.append(y)
            except Exception as e:
                print(f"插值 {method_name} 时出错: {e}")
                print(f"iterations: {iterations}")
                print(f"success_rates: {success_rates}")
                continue
        
        if not y_interp:
            print(f"方法 {method_name} 没有有效的插值结果")
            continue
            
        # 计算平均值和标准差
        y_mean = np.mean(y_interp, axis=0)
        y_std = np.std(y_interp, axis=0)
        
        # 绘制曲线和置信区间
        color = COLOR_MAP.get(method_name, None)
        plt.plot(x, y_mean, label=method_name, color=color)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
    
    # 添加图例和标签
    plt.xlabel('# of Training Steps (5e4 for Each Task)')
    plt.ylabel('Average Success Rate')
    plt.title('Performance across methods on the OCW10 sequence')
    plt.legend(loc='upper left', ncol=3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('performance_comparison.png', dpi=300)
    print("图像已保存为 performance_comparison.png")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    plot_performance_curves() 
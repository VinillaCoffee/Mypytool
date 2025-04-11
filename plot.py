import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
import seaborn as sns
import json
from matplotlib.ticker import FuncFormatter
import datetime
from scipy.signal import savgol_filter  # 添加Savitzky-Golay滤波器

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
    'finetuning': 'Finetuning',
    'None': 'Finetuning',  # 默认的Finetuning
    'None_fifo': 'Finetuning (FIFO)',  # 添加FIFO buffer的Finetuning
    'None_reservoir': 'Finetuning (Reservoir)'  # 添加Reservoir buffer的Finetuning
}

# 颜色映射，提高区分度
COLOR_MAP = {
    'PM': '#e6194B',             # 鲜红色
    'RWalk': '#f58231',          # 橙色
    'PackNet': '#3cb44b',        # 绿色
    'Finetuning': '#42d4f4',     # 青色
    'Finetuning (FIFO)': '#4363d8',      # 蓝色
    'Finetuning (Reservoir)': '#911eb4', # 紫色
    'L2': '#469990',             # 深青色
    'MAS': '#9A6324',            # 棕色
    'Prune': '#f032e6',          # 粉色
    'LWF': '#808000',            # 橄榄色
    'CoD': '#800000',            # 栗色
    'EWC': '#000075',            # 深蓝色
    'A-GEM': '#008080',          # 蓝绿色
    'VCL': '#e6beff',            # 淡紫色
    'Grow': '#ffe119'            # 黄色
}

def format_ticks(x, pos):
    """格式化大数字使其更易读 (例如, 1000 -> 1K, 1000000 -> 1M)"""
    if x >= 1e6:
        return f'{x*1e-6:.0f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    else:
        return f'{x:.0f}'

def extract_success_rates(tsv_path):
    """提取tsv文件中的成功率数据"""
    try:
        print(f"正在处理文件 {tsv_path}")
        
        # 使用pandas读取TSV文件
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            
            # 处理processed_progress.tsv格式
            if 'epoch' in df.columns and 'directory' in df.columns and 'method' in df.columns:
                print("检测到处理过的TSV文件格式")
                
                # 查找平均成功率列
                if 'test/stochastic/average_success' in df.columns:
                    success_col = 'test/stochastic/average_success'
                    print(f"使用随机策略的平均成功率列")
                elif 'test/deterministic/average_success' in df.columns:
                    success_col = 'test/deterministic/average_success'
                    print(f"使用确定性策略的平均成功率列")
                else:
                    # 查找任务特定的成功率列
                    stochastic_cols = [col for col in df.columns if 'success' in col and 'stochastic' in col]
                    if stochastic_cols:
                        success_col = stochastic_cols[0]
                        print(f"没有找到平均成功率列，使用第一个随机成功率列: {success_col}")
                    else:
                        print(f"在文件中没有找到任何成功率列")
                        return None, None
                
                # 提取迭代次数和成功率
                iterations = df['epoch'].values
                success_rates = df[success_col].values
                
                print(f"成功提取了 {len(iterations)} 个数据点")
                return iterations, success_rates
            
            # 检查是否包含必要的列
            if 'total_env_steps' not in df.columns:
                print(f"文件 {tsv_path} 中没有找到 'total_env_steps' 列")
                return None, None
            
            # 寻找包含success的列，优先使用average_success
            stochastic_cols = [col for col in df.columns if 'success' in col and 'stochastic' in col]
            deterministic_cols = [col for col in df.columns if 'success' in col and 'deterministic' in col]
            
            # 优先使用随机策略的平均成功率
            success_col = None
            for col in stochastic_cols:
                if 'average_success' in col:
                    success_col = col
                    break
            
            # 如果没有找到平均随机成功率，使用第一个随机成功率
            if success_col is None and stochastic_cols:
                success_col = stochastic_cols[0]
            
            # 如果没有随机成功率，尝试使用确定性策略
            if success_col is None:
                for col in deterministic_cols:
                    if 'average_success' in col:
                        success_col = col
                        break
                
                if success_col is None and deterministic_cols:
                    success_col = deterministic_cols[0]
            
            if success_col is None:
                print(f"在 {tsv_path} 中没有找到任何成功率列")
                return None, None
            
            print(f"使用成功率列: {success_col}")
            
            # 提取迭代次数和成功率
            iterations = df['total_env_steps'].values
            success_rates = df[success_col].values
            
            print(f"成功提取了 {len(iterations)} 个数据点")
            return iterations, success_rates
            
        except Exception as e:
            print(f"使用pandas读取文件失败: {e}")
            
            # 尝试直接按行读取文件
            with open(tsv_path, 'r') as f:
                lines = f.readlines()
            
            # 寻找列名行
            header_line = None
            for i, line in enumerate(lines):
                if 'total_env_steps' in line or 'epoch' in line:
                    header_line = i
                    break
            
            if header_line is None:
                print(f"在文件中没有找到header行")
                return None, None
            
            # 解析标题行
            headers = lines[header_line].strip().split('\t')
            success_idx = None
            steps_idx = None
            
            for i, h in enumerate(headers):
                if 'total_env_steps' in h or 'epoch' in h:
                    steps_idx = i
                if 'success' in h and ('average' in h or 'stochastic' in h):
                    success_idx = i
            
            if steps_idx is None or success_idx is None:
                print(f"在标题行中没有找到必要的列")
                return None, None
            
            # 解析数据行
            iterations = []
            success_rates = []
            
            for i in range(header_line + 1, len(lines)):
                try:
                    parts = lines[i].strip().split('\t')
                    if len(parts) > max(steps_idx, success_idx):
                        iterations.append(float(parts[steps_idx]))
                        success_rates.append(float(parts[success_idx]))
                except:
                    continue
            
            if not iterations:
                print(f"没有从文件中提取到有效数据")
                return None, None
            
            print(f"成功提取了 {len(iterations)} 个数据点")
            return np.array(iterations), np.array(success_rates)
    
    except Exception as e:
        print(f"处理 {tsv_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def smooth_curve(y, window=5, poly=3):
    """
    使用Savitzky-Golay滤波器平滑曲线
    
    参数:
    y: 需要平滑的数据
    window: 窗口大小，必须是奇数
    poly: 多项式阶数
    """
    if len(y) < window:
        return y
    return savgol_filter(y, window, poly)

def plot_performance_curves():
    """绘制所有方法的平均成功率曲线，仅使用seed=4的数据"""
    save_dir = 'cl'
    # 获取所有子文件夹
    directories = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    
    plt.figure(figsize=(12, 8))
    
    # 存储每个方法的数据，用于计算多个种子的平均值和标准差
    methods_data = {}
    
    # 限制处理的目录数量，用于测试
    test_mode = False
    if test_mode:
        print("测试模式：只处理前5个目录")
        directories = directories[:5]
    
    # 读取method_info.csv文件以获取方法信息
    method_info_path = "method_info.csv"
    if os.path.exists(method_info_path):
        try:
            method_df = pd.read_csv(method_info_path)
            print(f"已读取方法信息文件，包含 {len(method_df)} 个目录的方法信息")
            
            # 创建目录到方法和种子的映射
            dir_to_method = dict(zip(method_df['directory'], method_df['method']))
            dir_to_seed = dict(zip(method_df['directory'], method_df['seed']))
        except Exception as e:
            print(f"读取方法信息文件失败: {e}")
            dir_to_method = {}
            dir_to_seed = {}
    else:
        print(f"方法信息文件 {method_info_path} 不存在，将从config.json文件中读取方法信息")
        dir_to_method = {}
        dir_to_seed = {}
    
    for directory in directories:
        # 检查目录名是否符合cl_X的格式
        if not re.match(r'cl_\d+', directory):
            print(f"跳过不符合格式的目录: {directory}")
            continue
        
        # 从方法信息中获取seed
        seed = dir_to_seed.get(directory)
        
        # 如果没有从方法信息获取到seed，尝试从config.json获取
        if seed is None:
            config_path = os.path.join(save_dir, directory, 'config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    seed = config.get('seed')
                except Exception as e:
                    print(f"读取配置文件失败: {e}")
                    seed = None
        
        # 检查seed值，只处理seed=10的数据
        if seed != 4:
            print(f"跳过seed不是4的目录: {directory}, seed={seed}")
            continue
        
        # 获取方法名称
        method_key = dir_to_method.get(directory)
        
        # 如果从方法信息中没有获取到方法名称，尝试从config.json获取
        if method_key is None:
            config_path = os.path.join(save_dir, directory, 'config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # 获取方法名称，如果cl_method为null，则表示是基线方法
                    method_key = config.get('cl_method')
                    if method_key is None:
                        method_key = 'None'  # 将None作为键
                        
                        # 对于Finetuning方法（即method_key为None的情况），检查buffer类型
                        buffer_type = config.get('buffer_type', '').lower()  # 获取buffer类型，默认为空字符串
                        if buffer_type in ['fifo', 'reservoir']:
                            method_key = f'None_{buffer_type}'  # 例如：None_fifo 或 None_reservoir
                except Exception as e:
                    print(f"读取配置文件失败: {e}")
                    continue
            else:
                print(f"在 {directory} 中没有找到config.json文件")
                continue
        
        method_name = METHOD_NAMES.get(str(method_key), str(method_key))
        
        # 优先查找processed_progress.tsv文件
        tsv_path = os.path.join(save_dir, directory, 'processed_progress.tsv')
        if not os.path.exists(tsv_path):
            print(f"在 {directory} 中没有找到processed_progress.tsv文件，尝试查找原始的progress.tsv文件")
            tsv_path = os.path.join(save_dir, directory, 'progress.tsv')
            if not os.path.exists(tsv_path):
                print(f"在 {directory} 中也没有找到progress.tsv文件")
                continue
        
        # 提取成功率数据
        iterations, success_rates = extract_success_rates(tsv_path)
        if iterations is None or success_rates is None:
            continue
        
        print(f"提取 {method_name} 的数据: 迭代次数={len(iterations)}个, 成功率={len(success_rates)}个")
        
        # 存储数据
        if method_name not in methods_data:
            methods_data[method_name] = []
        
        # 存储(迭代次数, 成功率)对
        methods_data[method_name].append((iterations, success_rates))
    
    # 只有当有数据时才绘制图形
    if not methods_data:
        print("没有找到任何可用的数据，无法绘制图形")
        return
        
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
        
        # 对平均值进行平滑处理
        y_mean_smooth = smooth_curve(y_mean)
        
        # 绘制曲线和置信区间
        color = COLOR_MAP.get(method_name, None)
        plt.plot(x, y_mean_smooth, label=method_name, color=color, linewidth=1)  # 使用平滑后的曲线
        plt.fill_between(x, smooth_curve(y_mean - y_std), smooth_curve(y_mean + y_std), alpha=0.2, color=color)  # 平滑置信区间
    
    # 添加图例和标签
    plt.xlabel('epoch')
    plt.ylabel('Average Success')
    plt.title('Performance on the CW10 sequence')
    plt.legend(loc='upper left', ncol=3)  # 将图例移到右侧
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('cl_performance_comparison_seed4.png', dpi=300)  # 确保图例不被裁剪
    print("图像已保存为 cl_performance_comparison_seed4.png")
    
    # 显示图像
    plt.show()


def test_extract_single_file():
    """测试提取单个文件中的数据"""
    # 选择一个文件进行测试
    test_file = os.path.join('cl', 'cl_30', 'progress.tsv')
    if os.path.exists(test_file):
        iterations, success_rates = extract_success_rates(test_file)
        if iterations is not None and success_rates is not None:
            print(f"成功从 {test_file} 中提取了 {len(iterations)} 个数据点")
            # 绘制数据
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, success_rates, 'o-')
            plt.xlabel('Iterations')
            plt.ylabel('Success Rate')
            plt.title(f'Data from {test_file}')
            plt.grid(True)
            
            # 设置x轴刻度格式
            formatter = FuncFormatter(format_ticks)
            plt.gca().xaxis.set_major_formatter(formatter)
            
            plt.savefig('test_extraction.png')
            plt.show()
        else:
            print(f"无法从 {test_file} 中提取数据")
    else:
        print(f"测试文件 {test_file} 不存在")


def analyze_task_success(tsv_path, config_path, output_file=None):
    """分析任务成功率并按指定格式输出"""
    try:
        print(f"分析文件 {tsv_path} 中的任务成功率")
        
        # 读取配置文件获取方法名和种子
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 获取方法名
        method_key = config.get('cl_method')
        if method_key is None:
            method_key = 'None'
            buffer_type = config.get('buffer_type', '').lower()
            if buffer_type in ['fifo', 'reservoir']:
                method_key = f'None_{buffer_type}'
        
        method_name = METHOD_NAMES.get(str(method_key), str(method_key))
        seed = config.get('seed', 'unknown')
        
        # 生成类似"cw10-agem-MetaWorld-123-250318-110230"的标识符
        timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        identifier = f"cw{seed}-{method_name.lower()}-MetaWorld-{seed}-{timestamp}"
        
        # 读取TSV文件
        df = pd.read_csv(tsv_path, sep='\t')
        
        # 获取最后一行数据（最新的迭代）
        last_row = df.iloc[-1]
        
        # 获取总步数
        total_env_steps = last_row['total_env_steps']
        
        # 查找所有stochastic的success列
        success_cols = []
        return_cols = []
        task_names = []
        
        for col in df.columns:
            match = re.search(r'test/stochastic/(\d+)/([^/]+)/success', col)
            if match:
                task_index = match.group(1)
                task_name = match.group(2)
                success_cols.append(col)
                
                # 查找对应的return列
                return_col = f"test/stochastic/{task_index}/{task_name}/return/avg"
                if return_col in df.columns:
                    return_cols.append(return_col)
                else:
                    return_cols.append(None)
                
                task_names.append(task_name.replace('-v1', '-v2'))  # 将v1替换为v2以匹配示例格式
        
        # 准备输出
        output_lines = []
        success_values = []
        
        # 添加迭代次数行
        current_time = datetime.datetime.now().strftime("%y-%m-%d.%H:%M")
        output_lines.append(f"{current_time}|[{identifier}] Iteration {total_env_steps:>40}")
        
        # 添加每个任务的结果
        for i, (task, success_col, return_col) in enumerate(zip(task_names, success_cols, return_cols)):
            success_value = last_row[success_col]
            success_values.append(float(success_value))
            
            if return_col:
                return_value = last_row[return_col]
                output_lines.append(f"{current_time}|[{identifier}] test-seperate-{task}-Return {return_value:>20}")
            else:
                output_lines.append(f"{current_time}|[{identifier}] test-seperate-{task}-Return {0:>20}")
            
            output_lines.append(f"{current_time}|[{identifier}] test-seperate-{task}-Success {int(success_value):>20}")
        
        # 添加分隔线
        output_lines.append(f"{current_time}|[{identifier}] {'-'*42}  {'-'*11}")
        
        # 添加成功率列表
        output_lines.append(f"{current_time}|[{identifier}] {[success_values]}")
        
        # 输出到控制台
        for line in output_lines:
            print(line)
        
        # 如果指定了输出文件，写入文件
        if output_file:
            with open(output_file, 'w') as f:
                for line in output_lines:
                    f.write(line + "\n")
            print(f"结果已保存到 {output_file}")
        
        return success_values
    
    except Exception as e:
        print(f"分析任务成功率时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="绘制性能曲线并分析任务成功率")
    parser.add_argument("--analyze", action="store_true", help="是否分析任务成功率")
    parser.add_argument("--dir", type=str, default=None, help="要分析的特定目录，例如cl_10")
    parser.add_argument("--output", type=str, default=None, help="输出文件名")
    parser.add_argument("--plot", action="store_true", default=True, help="是否绘制性能曲线")
    args = parser.parse_args()
    
    if args.analyze:
        if args.dir:
            # 分析指定目录
            directory = args.dir
            save_dir = 'cl'
            
            config_path = os.path.join(save_dir, directory, 'config.json')
            tsv_path = os.path.join(save_dir, directory, 'progress.tsv')
            
            if os.path.exists(config_path) and os.path.exists(tsv_path):
                output_file = args.output if args.output else f"{directory}_success_analysis.txt"
                analyze_task_success(tsv_path, config_path, output_file)
            else:
                print(f"找不到配置文件或进度文件: {config_path} 或 {tsv_path}")
        else:
            print("请使用--dir参数指定要分析的目录")
    
    if args.plot:
        # 绘制性能曲线
        plot_performance_curves() 
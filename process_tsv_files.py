import os
import pandas as pd
import glob
import re
import numpy as np
from pathlib import Path
import json

def process_tsv_files():
    """
    处理cl文件夹下的所有TSV文件，将它们转换为类似cl_2/progress.tsv的格式
    """
    # 获取cl文件夹下的所有子文件夹
    cl_root = Path("cl")
    if not cl_root.exists():
        print(f"错误：找不到 {cl_root} 文件夹")
        return
    
    # 获取所有cl_X子文件夹
    cl_dirs = [d for d in cl_root.iterdir() if d.is_dir() and re.match(r'cl_\d+', d.name)]
    print(f"找到 {len(cl_dirs)} 个cl子文件夹")
    
    # 创建一个合并的DataFrame用于保存所有处理后的数据
    all_data = []
    method_info = []
    
    # 处理每个子文件夹中的progress.tsv文件
    for cl_dir in cl_dirs:
        progress_file = cl_dir / "progress.tsv"
        config_file = cl_dir / "config.json"
        
        if not progress_file.exists():
            print(f"警告：{progress_file} 文件不存在，跳过此目录")
            continue
        
        # 读取配置文件以获取方法信息
        method_name = "unknown"
        seed = "unknown"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # 获取方法名称
                method_key = config.get('cl_method')
                if method_key is None:
                    method_key = 'None'  # Finetuning
                    buffer_type = config.get('buffer_type', '').lower()
                    if buffer_type in ['fifo', 'reservoir']:
                        method_key = f'None_{buffer_type}'
                
                method_name = method_key
                seed = config.get('seed', 'unknown')
            except Exception as e:
                print(f"读取配置文件 {config_file} 时出错：{e}")
        
        try:
            print(f"处理文件：{progress_file}")
            
            # 读取TSV文件
            try:
                # 尝试使用pandas读取文件
                df = pd.read_csv(progress_file, sep='\t')
                
                # 提取cl_2格式的数据列
                extracted_data = extract_cl2_format(df, cl_dir.name, method_name, seed)
                if extracted_data is not None:
                    all_data.append(extracted_data)
                    method_info.append({
                        'directory': cl_dir.name,
                        'method': method_name,
                        'seed': seed
                    })
                
                # 保存处理后的文件
                output_file = cl_dir / "processed_progress.tsv"
                if extracted_data is not None:
                    extracted_data.to_csv(output_file, sep='\t', index=False)
                    print(f"成功处理并保存到：{output_file}")
                
            except Exception as e:
                print(f"使用pandas读取文件时出错：{e}")
                print("尝试手动处理文件...")
                
                # 手动读取和处理文件
                with open(progress_file, 'r') as f:
                    lines = f.readlines()
                
                # 手动处理逻辑
                header = lines[0].strip().split('\t')
                
                # 查找关键列
                success_indices = []
                train_success_idx = None
                train_active_env_idx = None
                epoch_idx = None
                stochastic_avg_success_idx = None
                deterministic_avg_success_idx = None
                
                for i, col in enumerate(header):
                    # 寻找所有任务的success列
                    if '/success' in col and 'stochastic' in col and 'average' not in col:
                        success_indices.append(i)
                    # 训练成功率
                    elif col == 'train/success':
                        train_success_idx = i
                    # 当前环境
                    elif col == 'train/active_env':
                        train_active_env_idx = i
                    # epoch
                    elif col == 'epoch':
                        epoch_idx = i
                    # 随机策略平均成功率
                    elif col == 'test/stochastic/average_success':
                        stochastic_avg_success_idx = i
                    # 确定性策略平均成功率
                    elif col == 'test/deterministic/average_success':
                        deterministic_avg_success_idx = i
                
                # 创建新的列名
                new_header = []
                
                for i, idx in enumerate(success_indices):
                    col_parts = header[idx].split('/')
                    if len(col_parts) >= 3:
                        task_id = col_parts[1]
                        task_name = col_parts[2]
                        new_header.append(f"test/stochastic/{task_id}/{task_name}/success")
                    else:
                        new_header.append(header[idx])
                
                # 添加其他列
                if stochastic_avg_success_idx is not None:
                    new_header.append('test/stochastic/average_success')
                if deterministic_avg_success_idx is not None:
                    new_header.append('test/deterministic/average_success')
                if epoch_idx is not None:
                    new_header.append('epoch')
                if train_success_idx is not None:
                    new_header.append('train/success')
                if train_active_env_idx is not None:
                    new_header.append('train/active_env')
                
                # 创建新的数据
                new_data = []
                new_data.append('\t'.join(new_header) + '\n')
                
                for line in lines[1:]:  # 跳过标题行
                    parts = line.strip().split('\t')
                    if len(parts) <= 1:
                        continue
                    
                    new_line = []
                    
                    # 添加成功率列
                    for idx in success_indices:
                        if idx < len(parts):
                            new_line.append(parts[idx])
                        else:
                            new_line.append('0.0')
                    
                    # 添加其他列
                    if stochastic_avg_success_idx is not None and stochastic_avg_success_idx < len(parts):
                        new_line.append(parts[stochastic_avg_success_idx])
                    elif len(success_indices) > 0:
                        # 如果没有平均值列，计算一个
                        success_values = [float(parts[idx]) for idx in success_indices if idx < len(parts)]
                        avg = sum(success_values) / len(success_values) if success_values else 0.0
                        new_line.append(str(avg))
                    
                    if deterministic_avg_success_idx is not None and deterministic_avg_success_idx < len(parts):
                        new_line.append(parts[deterministic_avg_success_idx])
                    else:
                        new_line.append('0.0')
                    
                    if epoch_idx is not None and epoch_idx < len(parts):
                        new_line.append(parts[epoch_idx])
                    else:
                        # 如果没有epoch列，使用行号作为epoch
                        new_line.append(str(lines.index(line)))
                    
                    if train_success_idx is not None and train_success_idx < len(parts):
                        new_line.append(parts[train_success_idx])
                    else:
                        new_line.append('0.0')
                    
                    if train_active_env_idx is not None and train_active_env_idx < len(parts):
                        new_line.append(parts[train_active_env_idx])
                    else:
                        new_line.append('0')
                    
                    new_data.append('\t'.join(new_line) + '\n')
                
                # 保存处理后的文件
                output_file = cl_dir / "processed_progress.tsv"
                with open(output_file, 'w') as f:
                    f.writelines(new_data)
                print(f"成功手动处理并保存到：{output_file}")
                
        except Exception as e:
            print(f"处理 {progress_file} 时发生错误：{e}")
    
    # 保存方法信息
    if method_info:
        method_df = pd.DataFrame(method_info)
        method_df.to_csv("method_info.csv", index=False)
        print("方法信息已保存到 method_info.csv")
    
    # 合并所有数据并保存
    if all_data:
        try:
            # 合并所有数据
            merged_data = pd.concat(all_data, ignore_index=True)
            
            # 保存合并后的数据
            merged_data.to_csv("all_progress.tsv", sep='\t', index=False)
            print("所有处理后的数据已合并保存到 all_progress.tsv")
        except Exception as e:
            print(f"合并数据时出错：{e}")
    
    print("处理完成！")

def extract_cl2_format(df, directory, method_name, seed):
    """提取类似cl_2格式的关键列"""
    try:
        # 创建一个新的DataFrame来存储提取的数据
        extracted_df = pd.DataFrame()
        
        # 提取所有成功率列
        success_cols = []
        for col in df.columns:
            if '/success' in col and 'stochastic' in col and 'average' not in col:
                extracted_df[col] = df[col]
                success_cols.append(col)
        
        # 计算并添加平均成功率列
        if 'test/stochastic/average_success' in df.columns:
            extracted_df['test/stochastic/average_success'] = df['test/stochastic/average_success']
        elif success_cols:
            extracted_df['test/stochastic/average_success'] = df[success_cols].mean(axis=1)
        
        # 添加确定性策略的平均成功率
        if 'test/deterministic/average_success' in df.columns:
            extracted_df['test/deterministic/average_success'] = df['test/deterministic/average_success']
        else:
            extracted_df['test/deterministic/average_success'] = 0.0
        
        # 添加epoch列
        if 'epoch' in df.columns:
            extracted_df['epoch'] = df['epoch']
        else:
            # 如果没有epoch列，使用行号作为epoch
            extracted_df['epoch'] = range(1, len(df) + 1)
        
        # 添加目录和方法信息（必须保留）
        extracted_df['directory'] = directory
        extracted_df['method'] = method_name
        extracted_df['seed'] = seed
        
        print(f"从 {directory} 中提取了 {len(success_cols)} 个任务的成功率")
        return extracted_df
    
    except Exception as e:
        print(f"提取 {directory} 的关键指标时出错：{e}")
        return None

if __name__ == "__main__":
    process_tsv_files() 
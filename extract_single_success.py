import os
import json
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import re

def extract_single_success(directory="single", target_seed=2, target_epoch=50):
    """
    读取指定目录下所有single_*/config.json文件，提取seed=4且epoch=50时的average success值
    
    Args:
        directory: 包含单任务文件夹的目录路径
        target_seed: 目标种子值
        target_epoch: 目标训练轮次
    
    Returns:
        task_success_dict: 包含每个任务的成功率的字典
        single_success: 一维数组，包含按文件名后缀排序的成功率
    """
    # 构建搜索路径
    search_pattern = os.path.join(directory, "single_*", "config.json")
    config_files = glob.glob(search_pattern)
    
    if not config_files:
        print(f"在 {directory} 目录下找不到匹配 'single_*/config.json' 的文件")
        return None, None
    
    print(f"找到 {len(config_files)} 个config.json文件")
    
    # 存储任务和成功率的字典以及文件夹信息
    task_folder_dict = {}
    
    # 读取每个config文件
    for config_path in config_files:
        try:
            folder_path = os.path.dirname(config_path)
            folder_name = os.path.basename(folder_path)
            
            # 获取文件夹后缀数字
            folder_suffix = re.search(r'single_(\d+)', folder_name)
            if folder_suffix:
                folder_num = int(folder_suffix.group(1))
            else:
                continue
                
            # 读取config.json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # 检查seed是否为目标值
            if config_data.get('seed') == target_seed:
                # 提取任务名称
                task_name = config_data.get('task')
                
                if task_name:
                    # 寻找对应的progress.tsv文件
                    progress_path = os.path.join(folder_path, "progress.tsv")
                    
                    if os.path.exists(progress_path):
                        # 读取progress.tsv
                        df = pd.read_csv(progress_path, sep='\t')
                        
                        # 找到指定epoch的行
                        epoch_row = df[df['epoch'] == target_epoch]
                        
                        if not epoch_row.empty:
                            # 提取average_success值
                            avg_success = epoch_row['test/stochastic/average_success'].values[0]
                            
                            # 存储结果，包括文件夹编号
                            task_folder_dict[folder_num] = {
                                'folder': folder_name,
                                'task': task_name,
                                'success_rate': avg_success
                            }
                            
                            print(f"已提取 {folder_name} (任务: {task_name}): 成功率 = {avg_success}")
                        else:
                            print(f"警告: 在 {progress_path} 中找不到epoch={target_epoch}的数据")
                    else:
                        print(f"警告: 找不到进度文件 {progress_path}")
                else:
                    print(f"警告: {config_path} 中未指定任务名称")
                    
        except Exception as e:
            print(f"处理 {config_path} 时出错: {e}")
    
    # 检查是否找到任何匹配数据
    if not task_folder_dict:
        print(f"未找到seed={target_seed}且epoch={target_epoch}的数据")
        return {}, np.array([])
    
    # 按文件夹编号排序
    sorted_folder_nums = sorted(task_folder_dict.keys())
    
    # 创建一维数组和任务顺序列表
    single_success = np.array([task_folder_dict[num]['success_rate'] for num in sorted_folder_nums])
    ordered_tasks = [task_folder_dict[num]['task'] for num in sorted_folder_nums]
    
    # 打印结果
    print("\n----- 提取的成功率数据 -----")
    print(f"找到 {len(task_folder_dict)} 个匹配的任务:")
    for i, folder_num in enumerate(sorted_folder_nums):
        info = task_folder_dict[folder_num]
        print(f"{i+1}. {info['folder']} - {info['task']}: {info['success_rate']}")
    
    print("\n按文件夹编号排序的任务:")
    for i, task in enumerate(ordered_tasks):
        print(f"{i+1}. {task}")
    
    print("\nsingle_success 数组内容:")
    print(single_success)
    
    # 返回任务字典和排序后的成功率数组
    result_dict = {
        "folder_task_dict": task_folder_dict,
        "ordered_tasks": ordered_tasks
    }
    
    return result_dict, single_success

def save_results(result_dict, success_array, output_file="single_success_results.json"):
    """
    保存结果到JSON文件
    
    Args:
        result_dict: 结果字典
        success_array: 成功率数组
        output_file: 输出文件路径
    """
    # 转换数字键为字符串以便JSON序列化
    folder_task_dict = {str(k): v for k, v in result_dict["folder_task_dict"].items()}
    
    output_data = {
        "folder_task_dict": folder_task_dict,
        "ordered_tasks": result_dict["ordered_tasks"],
        "single_success": success_array.tolist()
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    # 提取数据
    result_dict, single_success = extract_single_success()
    
    # 保存结果
    if result_dict:
        save_results(result_dict, single_success) 
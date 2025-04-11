import os
import json
import glob
from pathlib import Path
from collections import Counter

def analyze_config_files(directory="single"):
    """
    读取指定目录下所有single_*/config.json文件，统计seed和task信息
    
    Args:
        directory: 包含单任务文件夹的目录路径
    """
    # 构建搜索路径
    search_pattern = os.path.join(directory, "single_*", "config.json")
    config_files = glob.glob(search_pattern)
    
    if not config_files:
        print(f"在 {directory} 目录下找不到匹配 'single_*/config.json' 的文件")
        return
    
    print(f"找到 {len(config_files)} 个config.json文件")
    
    # 存储seed和task信息
    all_seeds = []
    all_tasks = []
    config_info = []
    
    # 读取每个config文件
    for file_path in config_files:
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
                
                # 提取文件夹名称（例如：single_1）
                folder_name = os.path.basename(os.path.dirname(file_path))
                
                # 提取seed和task信息
                seed = config_data.get('seed', None)
                task = config_data.get('task', None)
                
                if seed is not None:
                    all_seeds.append(seed)
                
                if task is not None:
                    all_tasks.append(task)
                
                # 存储配置信息
                config_info.append({
                    'folder': folder_name,
                    'seed': seed,
                    'task': task
                })
                    
        except Exception as e:
            print(f"读取 {file_path} 时出错: {e}")
    
    # 统计信息
    print("\n----- Seed 统计信息 -----")
    seed_counter = Counter(all_seeds)
    for seed, count in sorted(seed_counter.items()):
        print(f"Seed {seed}: {count} 个实例")
    
    print("\n----- Task 统计信息 -----")
    task_counter = Counter(all_tasks)
    for task, count in sorted(task_counter.items()):
        print(f"Task '{task}': {count} 个实例")
    
    # 输出总结
    print(f"\n总共分析了 {len(config_files)} 个config文件")
    print(f"包含 {len(seed_counter)} 个不同的seed值")
    print(f"包含 {len(task_counter)} 个不同的task")
    
    # 返回统计数据以便进一步处理
    return {
        "seed_counts": seed_counter,
        "task_counts": task_counter,
        "config_info": config_info
    }

def save_statistics(stats, output_file="single_config_statistics.json"):
    """
    将统计结果保存到JSON文件
    
    Args:
        stats: 包含统计信息的字典
        output_file: 输出文件路径
    """
    # 将Counter对象转换为普通字典
    result = {
        "seed_counts": {str(k): v for k, v in stats["seed_counts"].items()},
        "task_counts": {str(k): v for k, v in stats["task_counts"].items()},
        "config_info": stats["config_info"]
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"统计结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存统计结果时出错: {e}")

if __name__ == "__main__":
    # 分析配置文件
    stats = analyze_config_files()
    
    # 保存统计结果
    if stats:
        save_statistics(stats) 
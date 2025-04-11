import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

def analyze_configs():
    """分析所有cl_文件夹中的config.json文件，提取关键参数"""
    
    # 查找所有cl_文件夹
    cl_dirs = glob.glob('cl/cl_*')
    
    # 创建数据存储字典
    configs_data = []
    
    # 遍历每个文件夹
    for cl_dir in sorted(cl_dirs):
        # 获取cl_名称
        cl_name = os.path.basename(cl_dir)
        
        # 配置文件路径
        config_path = os.path.join(cl_dir, 'config.json')
        
        # 默认值
        buffer_type = "未指定"
        cl_method = "未指定"
        tasks = "未指定"
        additional_params = {}
        
        # 检查配置文件是否存在
        if os.path.exists(config_path):
            try:
                # 读取配置文件
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 提取主要参数
                buffer_type = config.get('buffer_type', "未指定")
                cl_method = config.get('cl_method', "未指定")
                tasks = config.get('tasks', "未指定")
                
                # 处理None值
                if buffer_type is None:
                    buffer_type = "None"
                if cl_method is None:
                    cl_method = "None"
                if tasks is None:
                    tasks = "None"
                
                # 提取其他可能感兴趣的参数
                for param in ['experiment_id', 'seed', 'steps_per_task', 'lr', 'batch_size', 
                             'reset_buffer_on_task_change', 'reset_optimizer_on_task_change',
                             'cl_reg_coef', 'hidden_sizes']:
                    if param in config:
                        additional_params[param] = config[param]
                
            except Exception as e:
                print(f"读取{config_path}时出错: {e}")
        else:
            print(f"文件{config_path}不存在")
        
        # 添加到数据列表
        entry = {
            "方法目录": cl_name,
            "buffer_type": buffer_type,
            "cl_method": cl_method,
            "tasks": tasks
        }
        # 添加其他参数
        entry.update(additional_params)
        configs_data.append(entry)
    
    # 转换为DataFrame
    df = pd.DataFrame(configs_data)
    
    # 如果没有数据，提前返回
    if len(df) == 0:
        print("未找到任何配置文件")
        return None
    
    # 计算各参数的统计信息
    stats = {
        "buffer_type": Counter(df["buffer_type"]),
        "cl_method": Counter(df["cl_method"]),
        "tasks": Counter(df["tasks"])
    }
    
    return df, stats

def print_stats(df, stats):
    """打印数据统计信息"""
    
    print("\n" + "="*80)
    print(f"共分析了 {len(df)} 个cl_目录中的config.json文件")
    print("="*80)
    
    # 打印所有数据
    print("\n完整参数列表:")
    print("="*80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
    
    # 打印buffer_type统计
    print("\n\nbuffer_type参数统计:")
    print("="*50)
    for buffer_type, count in stats["buffer_type"].most_common():
        print(f"{buffer_type}: {count}个实验 ({count/len(df)*100:.1f}%)")
    
    # 打印cl_method统计
    print("\ncl_method参数统计:")
    print("="*50)
    for cl_method, count in stats["cl_method"].most_common():
        print(f"{cl_method}: {count}个实验 ({count/len(df)*100:.1f}%)")
    
    # 打印tasks统计
    print("\ntasks参数统计:")
    print("="*50)
    for tasks, count in stats["tasks"].most_common():
        print(f"{tasks}: {count}个实验 ({count/len(df)*100:.1f}%)")
    
    # 输出详细的cl_method和buffer_type组合统计
    print("\ncl_method与buffer_type组合:")
    print("="*50)
    method_buffer_counts = df.groupby(['cl_method', 'buffer_type']).size().reset_index()
    method_buffer_counts.columns = ['cl_method', 'buffer_type', '实验数量']
    method_buffer_counts = method_buffer_counts.sort_values('实验数量', ascending=False)
    print(method_buffer_counts.to_string(index=False))

def save_to_excel(df):
    """将数据保存到Excel文件"""
    output_file = "cl_configs_analysis.xlsx"
    try:
        # 创建一个Excel Writer对象
        with pd.ExcelWriter(output_file) as writer:
            # 写入完整数据
            df.to_excel(writer, sheet_name='所有配置', index=False)
            
            # 写入buffer_type分组
            buffer_type_groups = df.groupby('buffer_type')['方法目录'].apply(list).reset_index()
            buffer_type_groups['方法数量'] = buffer_type_groups['方法目录'].apply(len)
            buffer_type_groups['方法目录'] = buffer_type_groups['方法目录'].apply(lambda x: ', '.join(x))
            buffer_type_groups = buffer_type_groups.sort_values('方法数量', ascending=False)
            buffer_type_groups.to_excel(writer, sheet_name='按buffer_type分组', index=False)
            
            # 写入cl_method分组
            cl_method_groups = df.groupby('cl_method')['方法目录'].apply(list).reset_index()
            cl_method_groups['方法数量'] = cl_method_groups['方法目录'].apply(len)
            cl_method_groups['方法目录'] = cl_method_groups['方法目录'].apply(lambda x: ', '.join(x))
            cl_method_groups = cl_method_groups.sort_values('方法数量', ascending=False)
            cl_method_groups.to_excel(writer, sheet_name='按cl_method分组', index=False)
            
            # 写入tasks分组
            tasks_groups = df.groupby('tasks')['方法目录'].apply(list).reset_index()
            tasks_groups['方法数量'] = tasks_groups['方法目录'].apply(len)
            tasks_groups['方法目录'] = tasks_groups['方法目录'].apply(lambda x: ', '.join(x))
            tasks_groups = tasks_groups.sort_values('方法数量', ascending=False)
            tasks_groups.to_excel(writer, sheet_name='按tasks分组', index=False)
            
            # 写入组合分组
            combo_groups = df.groupby(['cl_method', 'buffer_type'])['方法目录'].apply(list).reset_index()
            combo_groups['方法数量'] = combo_groups['方法目录'].apply(len)
            combo_groups['方法目录'] = combo_groups['方法目录'].apply(lambda x: ', '.join(x))
            combo_groups = combo_groups.sort_values('方法数量', ascending=False)
            combo_groups.to_excel(writer, sheet_name='cl_method与buffer_type组合', index=False)
            
        print(f"\n详细分析已保存到文件: {output_file}")
        return True
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        return False

def create_visualizations(df, stats):
    """创建可视化图表"""
    output_dir = "config_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    
    # 1. buffer_type饼图
    plt.figure(figsize=(10, 6))
    buffer_counts = df['buffer_type'].value_counts()
    plt.pie(buffer_counts, labels=buffer_counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05]*len(buffer_counts))
    plt.title('buffer_type分布', fontsize=15)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'buffer_type_pie.png'), dpi=300)
    plt.close()
    
    # 2. cl_method条形图
    plt.figure(figsize=(12, 8))
    method_counts = df['cl_method'].value_counts()
    colors = plt.cm.viridis(np.linspace(0, 1, len(method_counts)))
    bars = plt.bar(method_counts.index, method_counts.values, color=colors)
    plt.title('cl_method分布', fontsize=15)
    plt.xlabel('方法')
    plt.ylabel('实验数量')
    plt.xticks(rotation=45)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}个\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cl_method_bar.png'), dpi=300)
    plt.close()
    
    # 3. cl_method与buffer_type的交叉表热图
    plt.figure(figsize=(14, 8))
    crosstab = pd.crosstab(df['cl_method'], df['buffer_type'])
    sns.heatmap(crosstab, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': '实验数量'})
    plt.title('cl_method与buffer_type组合分布', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_buffer_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"\n可视化图表已保存到目录: {output_dir}")
    return True

def search_experiment(df, search_term):
    """根据关键词搜索实验"""
    search_term = search_term.lower()
    
    # 搜索方法目录中包含关键词的实验
    dir_matches = df[df['方法目录'].str.lower().str.contains(search_term)]
    
    # 搜索cl_method中包含关键词的实验
    method_matches = df[df['cl_method'].astype(str).str.lower().str.contains(search_term)]
    
    # 搜索buffer_type中包含关键词的实验
    buffer_matches = df[df['buffer_type'].astype(str).str.lower().str.contains(search_term)]
    
    # 合并结果并去重
    all_matches = pd.concat([dir_matches, method_matches, buffer_matches]).drop_duplicates()
    
    return all_matches

def interactive_mode():
    """交互式查询模式"""
    df, stats = analyze_configs()
    
    while True:
        print("\n" + "="*80)
        print("交互式查询模式 - 选择操作:")
        print("1. 查看所有配置")
        print("2. 按cl_method筛选")
        print("3. 按buffer_type筛选")
        print("4. 搜索实验")
        print("5. 创建可视化图表")
        print("6. 导出到Excel")
        print("0. 退出")
        print("="*80)
        
        choice = input("请输入选项编号: ")
        
        if choice == "1":
            print("\n所有配置:")
            print(df[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
        
        elif choice == "2":
            methods = sorted(df['cl_method'].unique())
            print("\n可选的cl_method:")
            for i, method in enumerate(methods):
                print(f"{i+1}. {method}")
            
            method_choice = input("请输入方法编号或输入方法名: ")
            try:
                idx = int(method_choice) - 1
                selected_method = methods[idx]
            except:
                selected_method = method_choice
            
            filtered = df[df['cl_method'] == selected_method]
            print(f"\n使用cl_method '{selected_method}'的实验 ({len(filtered)}个):")
            print(filtered[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
        
        elif choice == "3":
            buffer_types = sorted(df['buffer_type'].unique())
            print("\n可选的buffer_type:")
            for i, buffer in enumerate(buffer_types):
                print(f"{i+1}. {buffer}")
            
            buffer_choice = input("请输入buffer_type编号或输入buffer_type名: ")
            try:
                idx = int(buffer_choice) - 1
                selected_buffer = buffer_types[idx]
            except:
                selected_buffer = buffer_choice
            
            filtered = df[df['buffer_type'] == selected_buffer]
            print(f"\n使用buffer_type '{selected_buffer}'的实验 ({len(filtered)}个):")
            print(filtered[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
        
        elif choice == "4":
            search_term = input("\n请输入搜索关键词: ")
            results = search_experiment(df, search_term)
            print(f"\n搜索结果 ({len(results)}个):")
            if len(results) > 0:
                print(results[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
            else:
                print("未找到匹配的实验")
        
        elif choice == "5":
            create_visualizations(df, stats)
        
        elif choice == "6":
            save_to_excel(df)
        
        elif choice == "0":
            print("\n退出程序")
            break
        
        else:
            print("\n无效选项，请重新输入")

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
    
    # 读取每个config文件
    for file_path in config_files:
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
                
                # 提取seed和task信息
                if 'seed' in config_data:
                    all_seeds.append(config_data['seed'])
                
                if 'task' in config_data:
                    all_tasks.append(config_data['task'])
                    
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
        "task_counts": task_counter
    }

def save_statistics(stats, output_file="config_statistics.json"):
    """
    将统计结果保存到JSON文件
    
    Args:
        stats: 包含统计信息的字典
        output_file: 输出文件路径
    """
    # 将Counter对象转换为普通字典
    result = {
        "seed_counts": {str(k): v for k, v in stats["seed_counts"].items()},
        "task_counts": {str(k): v for k, v in stats["task_counts"].items()}
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"统计结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存统计结果时出错: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析cl_目录中的config.json文件配置参数')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='启用交互式查询模式')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='创建可视化图表')
    parser.add_argument('--search', '-s', type=str, default=None,
                        help='搜索特定实验')
    parser.add_argument('--method', '-m', type=str, default=None,
                        help='按cl_method筛选')
    parser.add_argument('--buffer', '-b', type=str, default=None,
                        help='按buffer_type筛选')
    
    args = parser.parse_args()
    
    print("开始分析cl_目录中的config.json文件...\n")
    
    # 分析配置
    result = analyze_configs()
    
    if result is not None:
        df, stats = result
        
        if args.interactive:
            interactive_mode()
        else:
            # 打印统计结果
            print_stats(df, stats)
            
            # 如果指定了搜索参数
            if args.search:
                print(f"\n搜索关键词 '{args.search}' 的结果:")
                results = search_experiment(df, args.search)
                print(f"找到 {len(results)} 个匹配的实验:")
                if len(results) > 0:
                    print(results[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
            
            # 如果指定了方法筛选参数
            if args.method:
                print(f"\n使用cl_method '{args.method}' 的实验:")
                filtered = df[df['cl_method'] == args.method]
                print(f"找到 {len(filtered)} 个匹配的实验:")
                if len(filtered) > 0:
                    print(filtered[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
            
            # 如果指定了buffer筛选参数
            if args.buffer:
                print(f"\n使用buffer_type '{args.buffer}' 的实验:")
                filtered = df[df['buffer_type'] == args.buffer]
                print(f"找到 {len(filtered)} 个匹配的实验:")
                if len(filtered) > 0:
                    print(filtered[['方法目录', 'buffer_type', 'cl_method', 'tasks']].to_string(index=False))
            
            # 创建可视化
            if args.visualize:
                create_visualizations(df, stats)
            
            # 保存到Excel
            save_to_excel(df)
    else:
        print("分析失败，未找到有效的配置文件")

    # 分析配置文件
    stats = analyze_config_files()
    
    # 保存统计结果
    if stats:
        save_statistics(stats) 
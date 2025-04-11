import os
import pandas as pd
from pathlib import Path
import re

def clean_processed_files():
    """
    处理所有processed_progress.tsv文件：
    1. 仅保留epoch <= 500的数据
    2. 删除多余的列
    3. 计算前10个任务(0-9)的平均成功率
    """
    cl_root = Path("cl")
    if not cl_root.exists():
        print(f"错误：找不到 {cl_root} 文件夹")
        return
    
    processed_files = list(cl_root.glob("**/processed_progress.tsv"))
    print(f"找到 {len(processed_files)} 个processed_progress.tsv文件")
    
    for file_path in processed_files:
        clean_single_file(file_path)
    
    # 处理合并文件
    all_progress_path = Path("all_progress.tsv")
    if all_progress_path.exists():
        print(f"处理合并文件 {all_progress_path}")
        clean_single_file(all_progress_path)
    
    print("处理完成！")

def clean_single_file(file_path):
    """处理单个processed_progress.tsv文件"""
    try:
        print(f"处理文件：{file_path}")
        
        # 读取TSV文件
        df = pd.read_csv(file_path, sep='\t')
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # 筛选epoch <= 500的数据
        if 'epoch' in df.columns:
            df = df[df['epoch'] <= 500]
            filtered_rows = len(df)
            print(f"  筛选前行数：{original_rows}，筛选后行数：{filtered_rows}")
        
        # 保留需要的列
        essential_cols = []
        
        # 只保留前10个任务的success列 (0-9)，并收集用于计算平均值的列
        first_10_tasks_success_cols = []
        task_pattern = re.compile(r'test/stochastic/(\d+)/.*?/success')
        
        for col in df.columns:
            match = task_pattern.match(col)
            if match:
                task_id = int(match.group(1))
                if task_id < 10:  # 只保留0-9号任务
                    essential_cols.append(col)
                    first_10_tasks_success_cols.append(col)
            elif '/success' in col and 'average' in col:
                # 保留平均成功率列（如果存在）
                essential_cols.append(col)
            elif 'epoch' in col or 'directory' in col or 'method' in col or 'seed' in col:
                # 保留其他必要的列
                essential_cols.append(col)
        
        # 始终计算并添加前10个任务的平均成功率
        if first_10_tasks_success_cols:
            # 删除旧的平均成功率列（如果存在）
            if 'test/stochastic/average_success' in df.columns:
                df = df.drop(columns=['test/stochastic/average_success'])
                
            # 添加新的平均成功率列
            df['test/stochastic/average_success'] = df[first_10_tasks_success_cols].mean(axis=1)
            
            # 确保这个列也被保留
            if 'test/stochastic/average_success' not in essential_cols:
                essential_cols.append('test/stochastic/average_success')
                
            print(f"  已计算前10个任务(0-9)的平均成功率")
        else:
            print(f"  警告：没有找到前10个任务的成功率列，无法计算平均值")
        
        # 只保留必要的列
        df = df[essential_cols]
        final_cols = len(df.columns)
        
        print(f"  列数变化：{original_cols} -> {final_cols}")
        
        # 保存处理后的文件
        df.to_csv(file_path, sep='\t', index=False)
        print(f"  成功保存处理后的文件：{file_path}")
        
    except Exception as e:
        print(f"处理 {file_path} 时出错：{e}")

if __name__ == "__main__":
    clean_processed_files() 
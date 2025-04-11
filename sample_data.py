import os
import pandas as pd
from pathlib import Path
import re

def sample_processed_files():
    """
    对所有processed_progress.tsv文件进行处理：
    1. 每50个epoch保留一行数据（即epoch=0, 50, 100, 150...）
    2. 将结果保存为CSV文件在原目录下
    """
    cl_root = Path("cl")
    if not cl_root.exists():
        print(f"错误：找不到 {cl_root} 文件夹")
        return
    
    processed_files = list(cl_root.glob("**/processed_progress.tsv"))
    print(f"找到 {len(processed_files)} 个processed_progress.tsv文件")
    
    for file_path in processed_files:
        sample_single_file(file_path)
    
    # 处理合并文件
    all_progress_path = Path("all_progress.tsv")
    if all_progress_path.exists():
        print(f"处理合并文件 {all_progress_path}")
        sample_single_file(all_progress_path)
    
    print("处理完成！")

def sample_single_file(file_path):
    """对单个processed_progress.tsv文件进行抽样处理"""
    try:
        print(f"处理文件：{file_path}")
        
        # 读取TSV文件
        df = pd.read_csv(file_path, sep='\t')
        original_rows = len(df)
        
        # 确保'epoch'列存在
        if 'epoch' not in df.columns:
            print(f"  警告：文件 {file_path} 中没有'epoch'列，无法处理")
            return
        
        # 按每50个epoch抽样
        sampled_df = df[df['epoch'] % 50 == 0]
        sampled_rows = len(sampled_df)
        
        # 为避免所有数据都被过滤，确保至少保留第一行和最后一行
        if sampled_rows == 0:
            print(f"  警告：抽样结果为空，将保留第一行和最后一行")
            sampled_df = pd.concat([df.iloc[[0]], df.iloc[[-1]]])
            sampled_rows = len(sampled_df)
        
        print(f"  抽样前行数：{original_rows}，抽样后行数：{sampled_rows}")
        
        # 生成CSV文件名（保持原文件名但改变扩展名和添加后缀）
        csv_path = file_path.parent / f"{file_path.stem}_sampled.csv"
        
        # 保存为CSV文件
        sampled_df.to_csv(csv_path, index=False)
        print(f"  成功保存抽样后的CSV文件：{csv_path}")
        
    except Exception as e:
        print(f"处理 {file_path} 时出错：{e}")

if __name__ == "__main__":
    sample_processed_files() 
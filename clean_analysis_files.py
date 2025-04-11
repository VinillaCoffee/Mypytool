import os
from pathlib import Path
import re

def clean_analysis_files():
    """
    删除cl文件夹中的分析结果文件
    """
    cl_root = Path("cl")
    if not cl_root.exists():
        print(f"错误：找不到 {cl_root} 文件夹")
        return
    
    # 定义要匹配的文件模式
    file_patterns = [
        r".*_matrix_analysis\.txt$",   # 匹配matrix_analysis.txt结尾的文件
        r".*_analysis\.txt$",          # 匹配analysis.txt结尾的文件
    ]
    
    # 编译正则表达式
    patterns = [re.compile(pattern) for pattern in file_patterns]
    
    # 统计删除的文件数量
    deleted_count = 0
    deleted_files = []
    
    # 遍历cl文件夹及其子文件夹
    for root, dirs, files in os.walk(cl_root):
        for file in files:
            file_path = Path(root) / file
            
            # 检查文件名是否匹配模式
            if any(pattern.match(file) for pattern in patterns):
                try:
                    # 删除文件
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    deleted_count += 1
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")
    
    print(f"\n总共删除了 {deleted_count} 个分析结果文件")
    
    # 如果删除的文件数量较多，可以不打印详细列表
    if deleted_count <= 20:
        print("删除的文件列表:")
        for file in deleted_files:
            print(f"  {file}")

if __name__ == "__main__":
    # 执行删除操作
    print("开始清理分析结果文件...")
    clean_analysis_files() 
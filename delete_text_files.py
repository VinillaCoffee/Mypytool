import os
from pathlib import Path
import re

def delete_text_files():
    """
    删除cl文件夹中的所有文本文件
    """
    cl_root = Path("cl")
    if not cl_root.exists():
        print(f"错误：找不到 {cl_root} 文件夹")
        return
    
    # 定义要删除的文本文件扩展名
    text_extensions = [".txt", ".log", ".md"]
    
    # 统计删除的文件数量
    deleted_count = 0
    deleted_files = []
    
    # 遍历cl文件夹及其子文件夹
    for root, dirs, files in os.walk(cl_root):
        for file in files:
            file_path = Path(root) / file
            
            # 检查扩展名
            if any(file.endswith(ext) for ext in text_extensions):
                try:
                    # 删除文件
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    deleted_count += 1
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")
    
    print(f"\n总共删除了 {deleted_count} 个文本文件")
    
    # 如果删除的文件数量较多，可以不打印详细列表
    if deleted_count <= 20:
        print("删除的文件列表:")
        for file in deleted_files:
            print(f"  {file}")

if __name__ == "__main__":
    # 确认删除操作
    confirm = input("确定要删除cl文件夹中的所有文本文件吗？(y/n): ")
    if confirm.lower() == 'y':
        delete_text_files()
    else:
        print("操作已取消") 
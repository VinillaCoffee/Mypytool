import os

def generate_file_tree(directory, indent=0):
    # 获取当前目录的相对路径
    relative_path = os.path.basename(directory)
    print(' ' * indent + f'|-- {relative_path}/')  # 打印当前目录

    # 遍历目录中的所有文件和文件夹
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # 如果是文件夹，递归调用
            generate_file_tree(item_path, indent + 4)  # 增加缩进
        else:
            # 如果是文件，打印文件名
            print(' ' * (indent + 4) + f'|-- {item}')  # 打印文件名

# 使用示例
directory_path = r'F:\papers_ai4ao\Continual RL'  # 替换为您的文件夹路径
generate_file_tree(directory_path)
import os

def count_files_in_directory(directory, root_directory):
    file_count = 0
    file_types = {}
    folder_info = []  # 用于存储文件夹信息

    # 遍历目录中的所有文件和文件夹
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            # 检查文件名是否包含括号
            if '(' in item and ')' in item:
                print(f"删除文件: {item_path}")  # 输出删除的文件路径
                os.remove(item_path)  # 删除文件
                continue  # 跳过后续处理

            file_count += 1
            # 获取文件扩展名
            ext = os.path.splitext(item)[1]
            if ext in file_types:
                file_types[ext] += 1
            else:
                file_types[ext] = 1
        elif os.path.isdir(item_path):
            # 如果是文件夹，递归调用
            sub_count, sub_types, sub_folder_info = count_files_in_directory(item_path, root_directory)
            file_count += sub_count
            for ext, count in sub_types.items():
                if ext in file_types:
                    file_types[ext] += count
                else:
                    file_types[ext] = count
            folder_info.extend(sub_folder_info)  # 合并子文件夹信息

    # 收集文件夹信息
    if file_count > 0:
        relative_path = os.path.relpath(directory, root_directory)
        folder_info.append((relative_path, file_count, file_types))

    return file_count, file_types, folder_info

# 使用示例
directory_path = r'D:\Downloads\Compressed'  # 替换为您的文件夹路径
_, _, all_folder_info = count_files_in_directory(directory_path, directory_path)

# 输出总文件数为2000的文件夹
print("总文件数为2000的文件夹:")
for folder, count, types in all_folder_info:
    if count == 2000:
        print(f"文件夹: {folder}, 总文件数: {count}")

# 输出总文件数不是2000的文件夹
print("\n总文件数不是2000的文件夹:")
for folder, count, types in all_folder_info:
    if count != 2000:
        print(f"文件夹: {folder}, 总文件数: {count}")
        print("文件格式及数量:")
        for ext, count in types.items():
            print(f"{ext}: {count}")
        print()  # 添加空行以便于阅读
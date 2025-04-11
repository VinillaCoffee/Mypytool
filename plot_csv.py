total_success = []

import pandas as pd
import os
import json
from pathlib import Path
import metric

def read_csv_success_data(file_path):
    """
    读取CSV文件中的success列数据，并存储到total_success列表中
    
    Args:
        file_path: CSV文件路径
    """
    global total_success
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return
    df = pd.read_csv(file_path)
    success_columns = [col for col in df.columns if 'success' in col and 'average' not in col]
    success_columns.sort(key=lambda x: int(x.split('/')[1].split('_')[0]) if x.split('/')[1].isdigit() else 
                                     int(x.split('/')[1]) if x.split('/')[1].isdigit() else -1)
    total_success = []
    
    for _, row in df.iterrows():
        success_values = [row[col] for col in success_columns]
        total_success.append(success_values)

    return total_success

def save_results(results_list, output_file="cl_performance_results.json"):
    """
    将结果保存到JSON文件
    
    Args:
        results_list: 包含性能结果的字典列表
        output_file: 保存结果的文件路径
    """
    try:
        # 如果文件已存在，先读取内容
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        # 添加新的结果
        if isinstance(results_list, list):
            existing_data.extend(results_list)
        else:
            existing_data.append(results_list)
            
        # 写入文件
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    # 默认的CSV文件路径,只读取在config.json中seed=4的csv文件
    ref_data = [
        0.85,
        0.71,
        0.8699999999999999,
        0.68,
        0.6,
        0.7899999999999999,
        0.8699999999999999,
        0.27,
        0.89,
        0.8
    ]
    all_results = []
    
    for seed in range(0, 159):
        # 读取config.json文件
        config_path = f"cl/cl_{seed}/config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config['seed'] == 4:
                default_csv_path = f"cl/cl_{seed}/processed_progress_sampled.csv"
                # 使用示例
                read_csv_success_data(default_csv_path)
                
                # 打印total_success的一些信息
                if total_success and len(total_success) > 0 and all(isinstance(row, list) for row in total_success):
                    print(f"\n处理 cl_{seed} 的数据:")
                    for i, row in enumerate(total_success[:10]):
                        print(f"{row}")
                    
                    print('------------ Final Performance -----------------')
                    print('------------------------------------------------')
                    print(f'Final performance is {metric.calculate_mean_performance(total_success)}')
                    print(f'Final epoch performance is {metric.calculate_mean_performance(total_success)[-1]}')
                    print(f'Final forgetting is {metric.calculate_forgetting(total_success)}')
                    print(f'Final backward transfer is {metric.calculate_backward_transfer(total_success)}')
                    print(f'Final forward transfer is {metric.calculate_ftw(total_success, ref_data)}')
                    print('------------------------------------------------')
                    
                    # 获取配置信息
                    buffer_type = config.get('buffer_type', "未指定")
                    cl_method = config.get('cl_method', "未指定")
                
                    # 创建一个字典来存储结果
                    results = {
                        'buffer_type': buffer_type,
                        'cl_method': cl_method,
                        'Final performance': metric.calculate_mean_performance(total_success),
                        'Final epoch performance': metric.calculate_mean_performance(total_success)[-1],
                        'Final forgetting': metric.calculate_forgetting(total_success),
                        'Final backward transfer': metric.calculate_backward_transfer(total_success),
                        'Final forward transfer': metric.calculate_ftw(total_success, ref_data)
                    }
                    
                    # 将结果添加到列表中
                    all_results.append(results)
                else:
                    print(f"警告: cl_{seed}的数据无效或为空")
        except FileNotFoundError:
            print(f"警告: 找不到 cl_{seed} 的配置文件")
        except Exception as e:
            print(f"处理 cl_{seed} 时出错: {e}")
    
    # 保存所有结果
    if all_results:
        save_results(all_results)
        print(f"已保存 {len(all_results)} 个结果")
    else:
        print("没有找到有效的结果数据")
    


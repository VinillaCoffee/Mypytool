import json
import pandas as pd
from collections import defaultdict

def read_performance_data(file_path="cl_performance_results.json"):
    """
    读取性能数据JSON文件
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"成功读取 {len(data)} 条数据记录")
        return data
    except Exception as e:
        print(f"读取数据文件时出错: {e}")
        return []

def process_method_data(data):
    """
    处理数据，按cl_method分组
    """
    # 创建方法与结果的映射
    method_data = {}
    
    # 处理None值和映射方法名
    method_map = {
        None: "Fine-tuning",
        "l2": "L2",
        "ewc": "EWC",
        "mas": "MAS",
        "vcl": "VCL",
        "packnet": "PackNet",
        "agem": "A-GEM"
    }
    
    # 由于数据中有重复记录，我们只取每种方法的最后一条
    for entry in data:
        method = entry.get('cl_method')
        buffer_type = entry.get('buffer_type')
        
        # 处理reservoir的情况
        if buffer_type == "reservoir" and method is None:
            method_name = "reservoir"
        else:
            method_name = method_map.get(method, str(method))
        
        # 收集性能指标
        performance = entry.get('Final epoch performance')
        forgetting = entry.get('Final forgetting')
        forward_transfer = entry.get('Final forward transfer')
        
        # 确保所有值都存在
        if all(v is not None for v in [performance, forgetting, forward_transfer]):
            # 直接使用最新的数据
            method_data[method_name] = {
                'performance': performance,
                'forgetting': forgetting,
                'forward_transfer': forward_transfer
            }
    
    return method_data

def create_table(method_data):
    """
    创建表格数据
    """
    # 表头
    columns = ['method', 'performance', 'forgetting', 'f. transfer']
    
    # 数据行
    rows = []
    
    # 按指定顺序添加方法
    methods_order = [
        "Fine-tuning", 
        "L2", 
        "EWC", 
        "MAS", 
        "VCL", 
        "PackNet", 
        "reservoir", 
        "A-GEM"
    ]
    
    for method in methods_order:
        if method in method_data:
            data = method_data[method]
            
            # 格式化数据，保留两位小数
            performance = f"{data['performance']:.2f}"
            forgetting = f"{data['forgetting']:.2f}"
            forward_transfer = f"{data['forward_transfer']:.2f}"
            
            rows.append([
                method, 
                performance, 
                forgetting, 
                forward_transfer
            ])
    
    # 创建DataFrame
    df = pd.DataFrame(rows, columns=columns)
    return df

def save_to_csv(df, output_file="method_performance_table.csv"):
    """
    将表格保存为CSV文件
    """
    try:
        df.to_csv(output_file, index=False)
        print(f"表格已保存到 {output_file}")
    except Exception as e:
        print(f"保存表格时出错: {e}")

if __name__ == "__main__":
    # 读取数据
    data = read_performance_data()
    
    if data:
        # 处理数据并获取每种方法的性能
        method_data = process_method_data(data)
        
        # 创建表格
        table = create_table(method_data)
        
        # 显示表格
        print("\n生成的表格:")
        print(table)
        
        # 保存为CSV
        save_to_csv(table)
    else:
        print("没有有效数据可处理") 
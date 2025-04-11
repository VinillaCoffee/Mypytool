import numpy as np
from extract_single_success import extract_single_success, save_results

def compute_average_success():
    """
    调用extract_single_success函数，读取10个随机种子(0-9)的结果，
    并对single_success数组取平均值
    
    Returns:
        avg_success: 平均成功率数组
        all_seeds_data: 包含所有种子数据的字典
    """
    # 设置目标种子和轮次
    seeds = list(range(10))  # 0-9的10个种子
    target_epoch = 50
    
    # 存储所有种子的成功率
    all_success_arrays = []
    all_seeds_data = {}
    
    # 对每个种子调用extract_single_success
    for seed in seeds:
        print(f"\n处理种子 {seed}:")
        result_dict, single_success = extract_single_success(
            directory="single", 
            target_seed=seed, 
            target_epoch=target_epoch
        )
        
        # 检查是否成功提取数据
        if result_dict and len(single_success) > 0:
            all_success_arrays.append(single_success)
            all_seeds_data[seed] = {
                "result_dict": result_dict,
                "single_success": single_success.tolist()
            }
            print(f"种子 {seed} 的成功率数组长度: {len(single_success)}")
        else:
            print(f"警告: 种子 {seed} 未能提取有效数据")
    
    # 检查是否有成功提取的数据
    if not all_success_arrays:
        print("错误: 没有从任何种子中提取到有效数据")
        return None, {}
    
    # 检查所有数组长度是否一致
    array_lengths = [len(arr) for arr in all_success_arrays]
    if len(set(array_lengths)) > 1:
        print(f"警告: 不同种子的成功率数组长度不一致: {array_lengths}")
        # 使用最小长度截断所有数组
        min_length = min(array_lengths)
        all_success_arrays = [arr[:min_length] for arr in all_success_arrays]
        print(f"已将所有数组截断到最小长度: {min_length}")
    
    # 计算平均值
    if all_success_arrays:
        stacked_arrays = np.stack(all_success_arrays)
        avg_success = np.mean(stacked_arrays, axis=0)
        std_success = np.std(stacked_arrays, axis=0)
        
        print("\n----- 平均成功率结果 -----")
        print(f"处理了 {len(all_success_arrays)} 个种子的数据")
        print(f"平均成功率数组长度: {len(avg_success)}")
        print("平均成功率数组内容:")
        print(avg_success)
        print("标准差数组内容:")
        print(std_success)
        
        # 保存结果
        output_data = {
            "average_success": avg_success.tolist(),
            "std_success": std_success.tolist(),
            "all_seeds_data": all_seeds_data,
            "seeds_processed": len(all_success_arrays),
            "array_length": len(avg_success)
        }
        
        # 保存结果到JSON文件
        try:
            import json
            with open("average_success_results.json", 'w') as f:
                json.dump(output_data, f, indent=4)
            print("平均成功率结果已保存到 average_success_results.json")
        except Exception as e:
            print(f"保存结果时出错: {e}")
        
        return avg_success, all_seeds_data
    else:
        return None, {}

if __name__ == "__main__":
    avg_success, _ = compute_average_success() 
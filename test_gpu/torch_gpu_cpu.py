import torch
import timeit
import matplotlib.pyplot as plt

'''
以矩阵A[10,n]和矩阵B[n,10]的乘法运算（分别在cpu和gpu上运行）来测试
'''
def cpu_gpu_compare(n):
    # 检查是否有可用的GPU
    device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    
    # 在CPU上创建张量
    cpu_a = torch.randn(10, n, device=device_cpu)
    cpu_b = torch.randn(n, 10, device=device_cpu)
    print(f"CPU张量设备: {cpu_a.device}, {cpu_b.device}")
    
    # 在GPU上创建张量
    gpu_a = torch.randn(10, n, device=device_gpu)
    gpu_b = torch.randn(n, 10, device=device_gpu)
    print(f"GPU张量设备: {gpu_a.device}, {gpu_b.device}")
    
    def cpu_run():
        c = torch.matmul(cpu_a, cpu_b)
        return c
    
    def gpu_run():
        c = torch.matmul(gpu_a, gpu_b)
        torch.cuda.synchronize()  # 确保GPU操作完成
        return c
    
    # 热身，避免将初始化时间计算在内
    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    # print('warmup:', cpu_time, gpu_time)
    
    # 正式计算10次，取平均值
    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    # print('run_time:', cpu_time, gpu_time)
    
    return cpu_time, gpu_time

# 测试不同大小的矩阵
n_list1 = range(1, 20000, 50)
n_list2 = range(20001, 100000, 1000)
n_list = list(n_list1) + list(n_list2)
time_cpu = []
time_gpu = []

for n in n_list:
    print(f"测试矩阵大小: {n}")
    t = cpu_gpu_compare(n)
    time_cpu.append(t[0])
    time_gpu.append(t[1])

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(n_list, time_cpu, color='red', label='CPU')
plt.plot(n_list, time_gpu, color='green', linewidth=1.0, linestyle='--', label='GPU')
plt.ylabel('耗时', fontproperties='SimHei', fontsize=20)
plt.xlabel('计算量', fontproperties='SimHei', fontsize=20)
plt.title('CPU和GPU计算力比较 (PyTorch)', fontproperties='SimHei', fontsize=30)
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('torch_gpu_cpu_comparison.png')
plt.show() 
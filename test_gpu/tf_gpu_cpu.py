import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
'''
以矩阵A[10,n]和矩阵B[n,10]的乘法运算（分别在cpu和gpu上运行）来测试，
'''
def cpu_gpu_compare(n):
    with tf.device('/cpu:0'):                    ##指定操作用cpu计算
        cpu_a = tf.random.normal([10,n])        ##生成符合高斯分布的随机数矩阵，通过改变n大小，增减计算量
        cpu_b = tf.random.normal([n,10])
    print(cpu_a.device,cpu_b.device)
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10,n])
        gpu_b = tf.random.normal([n,10])
    print(gpu_a.device,gpu_b.device)
    def cpu_run():
        with tf.device('/cpu:0'):               ##矩阵乘法，此操作采用cpu计算
            c = tf.matmul(cpu_a,cpu_b)
        return c
    def gpu_run():
        with tf.device('/gpu:0'):              ##矩阵乘法，此操作采用gpu计算
            c = tf.matmul(gpu_a,gpu_b)
        return c
    ##第一次计算需要热身，避免将初始化时间计算在内
    cpu_time = timeit.timeit(cpu_run,number=10)
    gpu_time = timeit.timeit(gpu_run,number=10)
    #print('warmup:',cpu_time,gpu_time)
    ##正式计算10次，取平均值
    cpu_time = timeit.timeit(cpu_run,number=10)
    gpu_time = timeit.timeit(gpu_run,number=10)
    #print('run_time:',cpu_time,gpu_time)
    return cpu_time,gpu_time

n_list1 = range(1,20000,50)
n_list2 = range(20001,100000,1000)
n_list = list(n_list1)+list(n_list2)
time_cpu =[]
time_gpu =[]
for n in n_list:
    t=cpu_gpu_compare(n)
    time_cpu.append(t[0])
    time_gpu.append(t[1])

plt.plot(n_list,time_cpu,color = 'red',label='cpu')
plt.plot(n_list,time_gpu,color='green',linewidth=1.0,linestyle='--',label='gpu')
plt.ylabel('耗时',fontproperties = 'SimHei',fontsize = 20)
plt.xlabel('计算量',fontproperties = 'SimHei',fontsize = 20)
plt.title('cpu和gpu计算力比较',fontproperties = 'SimHei',fontsize = 30)
plt.legend(loc='upper right')
plt.show()
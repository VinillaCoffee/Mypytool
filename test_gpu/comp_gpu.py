import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
import numpy as np

'''
Complex computations to test performance differences between GPUs
'''
def cpu_gpu_compare(n, iterations=10):
    # Check available GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPU devices: {gpus}")
    
    # Get GPU device names
    gpu_names = []
    for i, gpu in enumerate(gpus):
        with tf.device(f'/gpu:{i}'):
            # Create a small tensor to force execution on this device
            a = tf.ones((1, 1))
            # Get device information
            device_details = a.device
            if "GPU" in device_details:
                # Extract GPU name using TensorFlow's device info
                device_name = tf.config.experimental.get_device_details(gpu)['device_name']
                gpu_names.append(device_name)
    
    if len(gpu_names) >= 1:
        gpu0_name = gpu_names[0]
        print(f"GPU 0: {gpu0_name}")
    if len(gpu_names) >= 2:
        gpu1_name = gpu_names[1]
        print(f"GPU 1: {gpu1_name}")
    
    # Create test data with larger dimensions
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([n, n])
        cpu_b = tf.random.normal([n, n])
    
    with tf.device('/gpu:0'):
        gpu0_a = tf.random.normal([n, n])
        gpu0_b = tf.random.normal([n, n])
    
    # Check if a second GPU is available
    if len(gpus) > 1:
        with tf.device('/gpu:1'):
            gpu1_a = tf.random.normal([n, n])
            gpu1_b = tf.random.normal([n, n])
    
    # Define a more complex operation for thorough GPU testing
    def complex_compute(a, b):
        # Matrix multiplication
        c = tf.matmul(a, b)
        # Multiple matrix multiplications to increase computation load
        for _ in range(5):
            c = tf.matmul(c, c)
        # Add non-linear operations
        d = tf.nn.relu(c)
        # Additional matrix operation
        e = tf.matmul(d, d)
        return e
    
    def cpu_run():
        with tf.device('/cpu:0'):
            return complex_compute(cpu_a, cpu_b)
    
    def gpu0_run():
        with tf.device('/gpu:0'):
            return complex_compute(gpu0_a, gpu0_b)
    
    def gpu1_run():
        if len(gpus) > 1:
            with tf.device('/gpu:1'):
                return complex_compute(gpu1_a, gpu1_b)
        return None
    
    # Warm-up run
    print("Executing warm-up run...")
    cpu_result = cpu_run()
    gpu0_result = gpu0_run()
    if len(gpus) > 1:
        gpu1_result = gpu1_run()
    
    # Measure time over multiple runs
    print(f"Executing {iterations} test iterations...")
    
    cpu_times = []
    gpu0_times = []
    gpu1_times = []
    
    for i in range(iterations):
        # CPU test
        start = timeit.default_timer()
        _ = cpu_run()
        end = timeit.default_timer()
        cpu_times.append(end - start)
        
        # GPU 0 test
        start = timeit.default_timer()
        _ = gpu0_run()
        end = timeit.default_timer()
        gpu0_times.append(end - start)
        
        # GPU 1 test (if available)
        if len(gpus) > 1:
            start = timeit.default_timer()
            _ = gpu1_run()
            end = timeit.default_timer()
            gpu1_times.append(end - start)
    
    # Calculate statistics
    cpu_avg = np.mean(cpu_times)
    gpu0_avg = np.mean(gpu0_times)
    
    print(f"\nPerformance comparison (matrix dimension n={n}):")
    print(f"CPU average time: {cpu_avg:.6f}s")
    if len(gpu_names) >= 1:
        print(f"{gpu0_name} average time: {gpu0_avg:.6f}s")
        print(f"CPU/{gpu0_name} speed ratio: {cpu_avg/gpu0_avg:.2f}x")
    else:
        print(f"GPU 0 average time: {gpu0_avg:.6f}s")
        print(f"CPU/GPU 0 speed ratio: {cpu_avg/gpu0_avg:.2f}x")
    
    if len(gpus) > 1:
        gpu1_avg = np.mean(gpu1_times)
        if len(gpu_names) >= 2:
            print(f"{gpu1_name} average time: {gpu1_avg:.6f}s")
            print(f"CPU/{gpu1_name} speed ratio: {cpu_avg/gpu1_avg:.2f}x")
            print(f"{gpu0_name}/{gpu1_name} speed ratio: {gpu0_avg/gpu1_avg:.2f}x")
            
            if gpu0_avg > gpu1_avg:
                print(f"Conclusion: {gpu1_name} is {gpu0_avg/gpu1_avg:.2f}x faster than {gpu0_name}")
            else:
                print(f"Conclusion: {gpu0_name} is {gpu1_avg/gpu0_avg:.2f}x faster than {gpu1_name}")
        else:
            print(f"GPU 1 average time: {gpu1_avg:.6f}s")
            print(f"CPU/GPU 1 speed ratio: {cpu_avg/gpu1_avg:.2f}x")
            print(f"GPU 0/GPU 1 speed ratio: {gpu0_avg/gpu1_avg:.2f}x")
            
            if gpu0_avg > gpu1_avg:
                print(f"Conclusion: GPU 1 is {gpu0_avg/gpu1_avg:.2f}x faster than GPU 0")
            else:
                print(f"Conclusion: GPU 0 is {gpu1_avg/gpu0_avg:.2f}x faster than GPU 1")
        
        return cpu_avg, gpu0_avg, gpu1_avg, gpu_names
    
    return cpu_avg, gpu0_avg, None, gpu_names

# Test different matrix dimensions with higher density
n_values = [500,1000,2000,3000,4000,6000,8000,10000]
cpu_times = []
gpu0_times = []
gpu1_times = []
gpu_names = []

for n in n_values:
    print(f"\nTesting matrix dimension n = {n}")
    results = cpu_gpu_compare(n)
    
    cpu_times.append(results[0])
    gpu0_times.append(results[1])
    
    if len(results) > 2 and results[2] is not None:
        gpu1_times.append(results[2])
        
    # Store GPU names
    if len(gpu_names) == 0 and len(results) > 3:
        gpu_names = results[3]

# Create visualization charts
plt.figure(figsize=(15, 10))

# Set GPU labels based on actual device names
gpu0_label = f"GPU 0: {gpu_names[0]}" if len(gpu_names) >= 1 else "GPU 0"
gpu1_label = f"GPU 1: {gpu_names[1]}" if len(gpu_names) >= 2 else "GPU 1"

# Add GPU comparison information to figure title
if len(gpu_names) >= 2:
    main_title = f"GPU Performance Comparison: {gpu_names[0]} vs {gpu_names[1]}"
    plt.suptitle(main_title, fontsize=18, y=0.98)

# 1. CPU vs GPU runtime comparison
plt.subplot(2, 2, 1)
plt.plot(n_values, cpu_times, 'ro-', label='CPU', linewidth=1)
plt.plot(n_values, gpu0_times, 'go-', label=gpu0_label, linewidth=1)
if len(gpu1_times) > 0:
    plt.plot(n_values, gpu1_times, 'bo-', label=gpu1_label, linewidth=1)

plt.xlabel('Matrix Dimension (n)', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
plt.title('CPU vs GPU Runtime Comparison', fontsize=16)
plt.grid(True)
plt.legend()

# 2. GPU comparison only (for better visibility of differences)
plt.subplot(2, 2, 2)
plt.plot(n_values, gpu0_times, 'go-', label=gpu0_label, linewidth=1)
if len(gpu1_times) > 0:
    plt.plot(n_values, gpu1_times, 'bo-', label=gpu1_label, linewidth=1)

plt.xlabel('Matrix Dimension (n)', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
if len(gpu_names) >= 2:
    plt.title(f'{gpu_names[0]} vs {gpu_names[1]} Runtime Comparison', fontsize=16)
else:
    plt.title('GPU 0 vs GPU 1 Runtime Comparison', fontsize=16)
plt.grid(True)
plt.legend()

# 3. GPU acceleration ratio
plt.subplot(2, 2, 3)
cpu_gpu0_ratio = [cpu/gpu0 for cpu, gpu0 in zip(cpu_times, gpu0_times)]
plt.plot(n_values, cpu_gpu0_ratio, 'mo-', label=f'CPU/{gpu0_label}', linewidth=1)

if len(gpu1_times) > 0:
    cpu_gpu1_ratio = [cpu/gpu1 for cpu, gpu1 in zip(cpu_times, gpu1_times)]
    plt.plot(n_values, cpu_gpu1_ratio, 'co-', label=f'CPU/{gpu1_label}', linewidth=1)

plt.xlabel('Matrix Dimension (n)', fontsize=14)
plt.ylabel('Acceleration Ratio', fontsize=14)
plt.title('GPU Acceleration Ratio Compared to CPU', fontsize=16)
plt.grid(True)
plt.legend()

# 4. GPU-to-GPU performance ratio
if len(gpu1_times) > 0:
    plt.subplot(2, 2, 4)
    gpu_ratio = [gpu0/gpu1 for gpu0, gpu1 in zip(gpu0_times, gpu1_times)]
    if len(gpu_names) >= 2:
        ratio_label = f'{gpu_names[0]}/{gpu_names[1]}'
    else:
        ratio_label = 'GPU 0/GPU 1'
    plt.plot(n_values, gpu_ratio, 'ko-', label=ratio_label, linewidth=1)
    
    # Add horizontal line showing equal performance baseline
    plt.axhline(y=1.0, color='r', linestyle='--')
    
    plt.xlabel('Matrix Dimension (n)', fontsize=14)
    plt.ylabel('Time Ratio', fontsize=14)
    if len(gpu_names) >= 2:
        plt.title(f'{gpu_names[0]} vs {gpu_names[1]} Performance Ratio', fontsize=16)
        # Add annotation to explain the ratio
        if gpu_ratio[-1] > 1:
            plt.annotate(f"{gpu_names[1]} is faster", xy=(n_values[-1], gpu_ratio[-1]), 
                        xytext=(n_values[-1]*0.8, gpu_ratio[-1]*1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        else:
            plt.annotate(f"{gpu_names[0]} is faster", xy=(n_values[-1], gpu_ratio[-1]), 
                        xytext=(n_values[-1]*0.8, gpu_ratio[-1]*0.9),
                        arrowprops=dict(facecolor='black', shrink=0.05))
    else:
        plt.title('GPU 0 vs GPU 1 Performance Ratio', fontsize=16)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
if len(gpu_names) >= 2:
    plt.subplots_adjust(top=0.92)  # Adjust for suptitle
plt.savefig('gpu_performance_comparison.png', dpi=300)
plt.show()

# Print final conclusion
if len(gpu1_times) > 0:
    final_ratio = gpu0_times[-1] / gpu1_times[-1]
    print("\nFinal Performance Analysis (at maximum matrix dimension):")
    if len(gpu_names) >= 2:
        if final_ratio > 1.05:
            print(f"{gpu_names[1]} is {final_ratio:.2f}x faster than {gpu_names[0]}")
        elif final_ratio < 0.95:
            print(f"{gpu_names[0]} is {1/final_ratio:.2f}x faster than {gpu_names[1]}")
        else:
            print(f"{gpu_names[0]} and {gpu_names[1]} have similar performance")
    else:
        if final_ratio > 1.05:
            print(f"GPU 1 is {final_ratio:.2f}x faster than GPU 0")
        elif final_ratio < 0.95:
            print(f"GPU 0 is {1/final_ratio:.2f}x faster than GPU 1")
        else:
            print("GPU 0 and GPU 1 have similar performance")
else:
    print("\nOnly one GPU detected, cannot compare multiple GPU performance")
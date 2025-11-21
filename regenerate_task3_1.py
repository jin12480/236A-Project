"""Regenerate Task 3.1 plot"""
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from MySolution import MyDecentralized, MyFeatureCompression

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Regenerating Task 3.1 plot...")
print("=" * 60)

# Load data
print("\nLoading datasets...")
mnist_data = prepare_mnist_data()
print("   [OK] Data loaded")

# Split into quadrants
print("\nSplitting data into 4 sensor quadrants...")
_, train_blocks = split_into_quadrants(mnist_data['trainX'])
_, val_blocks = split_into_quadrants(mnist_data['valX'])
_, test_blocks = split_into_quadrants(mnist_data['testX'])
print("   [OK] Blocks created")

# Get baseline
print("\nComputing Task 1 baseline...")
clf_mnist = MyDecentralized(K=3)
clf_mnist.train(mnist_data['trainX'], mnist_data['trainY'])
task1_mnist_acc = clf_mnist.evaluate(mnist_data['testX'], mnist_data['testY'])
print(f"   [OK] Baseline accuracy: {task1_mnist_acc:.4f}")

# Run Task 3.1
print("\nRunning Task 3.1: Decentralized per-sensor budget...")
fc = MyFeatureCompression(K=3)
k_list = [25, 100, 196, 392, 588, 784]

result_k = fc.run_decentralized_per_sensor(
    train_blocks, val_blocks, test_blocks,
    mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
    k_list=k_list
)

# Add baseline
result_k['baseline_test_accuracy_task1'] = task1_mnist_acc

print("\nResults:")
for k, acc, b_s in zip(result_k['k'], result_k['test_accuracy'], result_k['b_s']):
    print(f"  k={k:3d} bits/sensor, b_s={b_s}, accuracy={acc:.4f}")

# Generate plot
print("\nGenerating plot...")
plot_result_per_sensor(result3_1=result_k, ylim_lo=0.3)
plt.savefig('task3_1_per_sensor.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task3_1_per_sensor.png")

print("\n" + "=" * 60)
print("Task 3.1 plot regenerated successfully!")
print("=" * 60)


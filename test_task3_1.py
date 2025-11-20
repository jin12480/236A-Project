"""Test script for Task 3.1 - Decentralized Compression (Fixed Per-Sensor Budget)"""
from utils import prepare_mnist_data, split_into_quadrants
from MySolution import MyFeatureCompression, MyDecentralized
import numpy as np

print("=" * 60)
print("Testing Task 3.1: Decentralized Compression (Per-Sensor Budget)")
print("=" * 60)

# Load MNIST data
print("\nLoading Fashion-MNIST data...")
mnist_data = prepare_mnist_data()
print(f"[OK] Data loaded:")
print(f"  Train: {mnist_data['trainX'].shape}")
print(f"  Val:   {mnist_data['valX'].shape}")
print(f"  Test:  {mnist_data['testX'].shape}")

# Split into quadrants (sensors)
print("\nSplitting data into 4 sensor quadrants...")
_, train_blocks = split_into_quadrants(mnist_data['trainX'])
_, val_blocks = split_into_quadrants(mnist_data['valX'])
_, test_blocks = split_into_quadrants(mnist_data['testX'])

print(f"[OK] Blocks created:")
for i, block in enumerate(train_blocks):
    print(f"  Sensor {i+1}: {block.shape}")

# Get baseline from Task 1
print("\nComputing Task 1 baseline (no compression)...")
clf_baseline = MyDecentralized(K=3)
clf_baseline.train(mnist_data['trainX'], mnist_data['trainY'])
baseline_acc = clf_baseline.evaluate(mnist_data['testX'], mnist_data['testY'])
print(f"[OK] Baseline test accuracy: {baseline_acc:.4f}")

# Test Task 3.1
print("\nTesting decentralized compression with fixed per-sensor budget...")
fc = MyFeatureCompression(K=3)

# Test with a few per-sensor budgets
k_list = [25, 100, 196, 392, 588, 784]
print(f"\nTesting per-sensor budgets: {k_list}")
print("(This may take a few minutes...)")

result = fc.run_decentralized_per_sensor(
    train_blocks, val_blocks, test_blocks,
    mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
    k_list=k_list
)

# Add baseline to result for plotting
result['baseline_test_accuracy_task1'] = baseline_acc

print("\n" + "=" * 60)
print("Task 3.1 Results:")
print("=" * 60)
print(f"Baseline (Task 1, no compression): {baseline_acc:.4f}")
print("\nPer-sensor budget results:")
for i, (k, acc, b_s) in enumerate(zip(result['k'], result['test_accuracy'], result['b_s'])):
    total_bits = k * 4  # 4 sensors
    print(f"k={k:3d} bits/sensor (Total={total_bits:4d} bits, b_s={b_s}): Test Accuracy = {acc:.4f}")

print("\n" + "=" * 60)
print("Task 3.1 Testing Complete!")
print("=" * 60)


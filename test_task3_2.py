"""Test script for Task 3.2 - Decentralized Compression (Fixed Total Budget)"""
from utils import prepare_mnist_data, split_into_quadrants
from MySolution import MyFeatureCompression
import numpy as np

print("=" * 60)
print("Testing Task 3.2: Decentralized Compression (Total Budget)")
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

# Test Task 3.2
print("\nTesting decentralized compression with fixed total budget...")
fc = MyFeatureCompression(K=3)

# Test with a few total budgets
B_tot_list = [100, 400, 784, 1568, 2352, 3136]
print(f"\nTesting total budgets: {B_tot_list}")
print("(This may take several minutes...)")

result = fc.run_decentralized_total(
    train_blocks, val_blocks, test_blocks,
    mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
    B_tot_list=B_tot_list
)

print("\n" + "=" * 60)
print("Task 3.2 Results:")
print("=" * 60)
for i, (B_tot, acc, alloc) in enumerate(zip(result['B_tot'], result['test_accuracy'], result['best_allocation'])):
    actual_budget = sum(196 * b for b in alloc)  # Each sensor has 196 features
    print(f"Budget: {B_tot:5d} bits (alloc={alloc}, actual={actual_budget:5d}) -> Test Accuracy: {acc:.4f}")

print("\n" + "=" * 60)
print("Task 3.2 Testing Complete!")
print("=" * 60)


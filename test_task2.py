"""Test script for Task 2 - Centralized Compression"""
from utils import prepare_mnist_data
from MySolution import MyFeatureCompression
import numpy as np

print("=" * 60)
print("Testing Task 2: Centralized Compression")
print("=" * 60)

# Load MNIST data
print("\nLoading Fashion-MNIST data...")
mnist_data = prepare_mnist_data()
print(f"[OK] Data loaded:")
print(f"  Train: {mnist_data['trainX'].shape}")
print(f"  Val:   {mnist_data['valX'].shape}")
print(f"  Test:  {mnist_data['testX'].shape}")

# Test Task 2
print("\nTesting centralized compression...")
fc = MyFeatureCompression(K=3)

# Test with a few budgets
B_tot_list = [100, 400, 784, 1568, 2352, 3136]
print(f"\nTesting budgets: {B_tot_list}")
print("(This may take a few minutes...)")

result = fc.run_centralized(
    mnist_data['trainX'], mnist_data['trainY'],
    mnist_data['valX'],   mnist_data['valY'],
    mnist_data['testX'],  mnist_data['testY'],
    B_tot_list=B_tot_list
)

print("\n" + "=" * 60)
print("Task 2 Results:")
print("=" * 60)
for i, (B_tot, acc) in enumerate(zip(result['B_tot'], result['test_accuracy'])):
    b_per_feature = B_tot / 784
    print(f"Budget: {B_tot:5d} bits ({b_per_feature:.2f} bits/feature) -> Test Accuracy: {acc:.4f}")

print("\n" + "=" * 60)
print("Task 2 Testing Complete!")
print("=" * 60)


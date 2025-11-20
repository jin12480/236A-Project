"""Test script for Task 3.3 - Minimal Bits for Target Accuracy"""
from utils import prepare_mnist_data, split_into_quadrants
from MySolution import MyFeatureCompression, MyTargetAllocator
import numpy as np

print("=" * 60)
print("Testing Task 3.3: Minimal Bits for Target Accuracy")
print("=" * 60)

# Load MNIST data
print("\nLoading Fashion-MNIST data...")
mnist_data = prepare_mnist_data()
print(f"[OK] Data loaded")

# Split into quadrants for decentralized
print("\nSplitting data into 4 sensor quadrants...")
_, train_blocks = split_into_quadrants(mnist_data['trainX'])
_, val_blocks = split_into_quadrants(mnist_data['valX'])
_, test_blocks = split_into_quadrants(mnist_data['testX'])
print(f"[OK] Blocks created")

# Initialize
fc = MyFeatureCompression(K=3)
ta = MyTargetAllocator(K=3)

# Test with different target accuracies
alphas = [0.6, 0.7, 0.8, 0.9, 0.95]
B_grid = [100, 400, 784, 1568, 2352, 3136]

print(f"\nTesting target accuracies: {alphas}")
print(f"Searching over budgets: {B_grid}")
print("(This may take several minutes...)")

cent_min = []
decz_min = []

for alpha in alphas:
    print(f"\n--- Target accuracy alpha = {alpha} ---")
    
    # Centralized
    print("  Searching centralized...")
    Bc = ta.minimal_bits_centralized(
        fc, 
        mnist_data['trainX'], mnist_data['trainY'],
        mnist_data['valX'],   mnist_data['valY'],
        mnist_data['testX'],  mnist_data['testY'],
        alpha=alpha, B_grid=B_grid
    )
    cent_min.append(Bc)
    print(f"    Minimal bits (centralized): {Bc}")
    
    # Decentralized
    print("  Searching decentralized...")
    Bd, alloc = ta.minimal_bits_decentralized(
        fc,
        train_blocks, val_blocks, test_blocks,
        mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
        alpha=alpha, B_grid=B_grid
    )
    decz_min.append(Bd)
    print(f"    Minimal bits (decentralized): {Bd}, allocation: {alloc}")

# Create result dictionary
result_target = {
    'alphas': alphas,
    'centralized_min_bits': cent_min,
    'decentralized_min_bits': decz_min
}

print("\n" + "=" * 60)
print("Task 3.3 Results:")
print("=" * 60)
print(f"{'α':>6} | {'Centralized':>15} | {'Decentralized':>15} | {'Allocation':>20}")
print("-" * 60)
for a, Bc, Bd in zip(alphas, cent_min, decz_min):
    alloc_str = str(alloc) if Bd is not None else "N/A"
    Bc_str = str(Bc) if Bc is not None else "N/A"
    Bd_str = str(Bd) if Bd is not None else "N/A"
    print(f"{a:6.2f} | {Bc_str:>15} | {Bd_str:>15} | {alloc_str:>20}")

print("\n" + "=" * 60)
print("Task 3.3 Testing Complete!")
print("=" * 60)


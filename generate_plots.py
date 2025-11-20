"""Generate all required plots for the project report"""
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from MySolution import MyDecentralized, MyFeatureCompression, MyTargetAllocator

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Generating all plots for the project report...")
print("=" * 60)

# Load data
print("\n1. Loading datasets...")
syn_data = prepare_synthetic_data()
mnist_data = prepare_mnist_data()
print("   [OK] Datasets loaded")

# ============================================================================
# Task 1: Baseline Classifier
# ============================================================================
print("\n2. Task 1: Training baseline classifiers...")

# Synthetic dataset
clf_syn = MyDecentralized(K=3)
clf_syn.train(syn_data['trainX'], syn_data['trainY'])
task1_syn_acc = clf_syn.evaluate(syn_data['testX'], syn_data['testY'])

# Fashion-MNIST dataset
clf_mnist = MyDecentralized(K=3)
clf_mnist.train(mnist_data['trainX'], mnist_data['trainY'])
task1_mnist_acc = clf_mnist.evaluate(mnist_data['testX'], mnist_data['testY'])

print(f"   Synthetic: {task1_syn_acc:.4f}")
print(f"   Fashion-MNIST: {task1_mnist_acc:.4f}")

# Plot Task 1 results
plt.figure(figsize=(8, 6))
datasets = ['Synthetic\n(2D)', 'Fashion-MNIST\n(784D)']
accuracies = [task1_syn_acc, task1_mnist_acc]
colors = ['#3498db', '#e74c3c']
bars = plt.bar(datasets, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
plt.title('Task 1: Baseline Classifier Performance', fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('task1_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task1_baseline.png")

# ============================================================================
# Task 2: Centralized Compression
# ============================================================================
print("\n3. Task 2: Running centralized compression...")
fc = MyFeatureCompression(K=3)
B_tot_list = [100, 400, 784, 1568, 2352, 3136]

result_central = fc.run_centralized(
    mnist_data['trainX'], mnist_data['trainY'],
    mnist_data['valX'],   mnist_data['valY'],
    mnist_data['testX'],  mnist_data['testY'],
    B_tot_list=B_tot_list
)

# Plot Task 2
plt.figure(figsize=(10, 6))
plt.plot(result_central['B_tot'], result_central['test_accuracy'], 
         marker='o', markersize=10, linewidth=2.5, label='Centralized (Quantized)', color='#2ecc71')
plt.axhline(y=task1_mnist_acc, color='#e74c3c', linestyle='--', linewidth=2, 
            label=f'Task 1 Baseline ({task1_mnist_acc:.3f})')
plt.xlabel('Total Budget $B_{tot}$ (bits/image)', fontsize=12, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
plt.title('Task 2: Centralized Compression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig('task2_centralized.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task2_centralized.png")

# ============================================================================
# Task 3.1: Decentralized Per-Sensor Budget
# ============================================================================
print("\n4. Task 3.1: Running decentralized per-sensor compression...")
_, train_blocks = split_into_quadrants(mnist_data['trainX'])
_, val_blocks = split_into_quadrants(mnist_data['valX'])
_, test_blocks = split_into_quadrants(mnist_data['testX'])

k_list = [25, 100, 196, 392, 588, 784]
result_k = fc.run_decentralized_per_sensor(
    train_blocks, val_blocks, test_blocks,
    mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
    k_list=k_list
)
result_k['baseline_test_accuracy_task1'] = task1_mnist_acc

# Plot Task 3.1
plot_result_per_sensor(result3_1=result_k, ylim_lo=0.3)
plt.savefig('task3_1_per_sensor.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task3_1_per_sensor.png")

# ============================================================================
# Task 3.2: Decentralized Total Budget
# ============================================================================
print("\n5. Task 3.2: Running decentralized total budget compression...")
result_B = fc.run_decentralized_total(
    train_blocks, val_blocks, test_blocks,
    mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
    B_tot_list=B_tot_list
)

# Plot Task 3.2
plot_result_total(result2=None, result3_2=result_B, ylim_lo=0.3)
plt.savefig('task3_2_total_budget.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task3_2_total_budget.png")

# Plot Centralized vs Decentralized comparison
res_compare = {
    'centralized': result_central,
    'decentralized': result_B,
}
plot_result_centralized_vs_decentralized(result=res_compare, ylim_lo=0.3)
plt.savefig('task3_2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task3_2_comparison.png")

# ============================================================================
# Task 3.3: Minimal Bits for Target Accuracy
# ============================================================================
print("\n6. Task 3.3: Finding minimal bits for target accuracy...")
ta = MyTargetAllocator(K=3)
alphas = [0.6, 0.7, 0.8, 0.9, 0.95]
B_grid = [100, 400, 784, 1568, 2352, 3136]

cent_min = []
decz_min = []
decz_alloc = []

for alpha in alphas:
    print(f"   Processing alpha = {alpha}...")
    Bc = ta.minimal_bits_centralized(
        fc, mnist_data['trainX'], mnist_data['trainY'],
        mnist_data['valX'],   mnist_data['valY'],
        mnist_data['testX'],  mnist_data['testY'],
        alpha=alpha, B_grid=B_grid
    )
    Bd, alloc = ta.minimal_bits_decentralized(
        fc, train_blocks, val_blocks, test_blocks,
        mnist_data['trainY'], mnist_data['valY'], mnist_data['testY'],
        alpha=alpha, B_grid=B_grid
    )
    cent_min.append(Bc)
    decz_min.append(Bd)
    decz_alloc.append(alloc)

res_target = {
    'alphas': alphas,
    'centralized_min_bits': cent_min,
    'decentralized_min_bits': decz_min
}

# Plot Task 3.3
plot_result_target(result=res_target)
plt.savefig('task3_3_target_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: task3_3_target_accuracy.png")

print("\n" + "=" * 60)
print("All plots generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  - task1_baseline.png")
print("  - task2_centralized.png")
print("  - task3_1_per_sensor.png")
print("  - task3_2_total_budget.png")
print("  - task3_2_comparison.png")
print("  - task3_3_target_accuracy.png")


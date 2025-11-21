"""Generate remaining plots (Task 3.2 and 3.3)"""
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from MySolution import MyDecentralized, MyFeatureCompression, MyTargetAllocator

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Generating remaining plots...")
print("=" * 60)

# Load data
print("\nLoading datasets...")
mnist_data = prepare_mnist_data()
_, train_blocks = split_into_quadrants(mnist_data['trainX'])
_, val_blocks = split_into_quadrants(mnist_data['valX'])
_, test_blocks = split_into_quadrants(mnist_data['testX'])

# Get baseline
clf_mnist = MyDecentralized(K=3)
clf_mnist.train(mnist_data['trainX'], mnist_data['trainY'])
task1_mnist_acc = clf_mnist.evaluate(mnist_data['testX'], mnist_data['testY'])

fc = MyFeatureCompression(K=3)
B_tot_list = [100, 400, 784, 1568, 2352, 3136]

# ============================================================================
# Task 3.2: Decentralized Total Budget
# ============================================================================
print("\nTask 3.2: Running decentralized total budget compression...")
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

# Get centralized results for comparison
print("\nGetting centralized results for comparison...")
result_central = fc.run_centralized(
    mnist_data['trainX'], mnist_data['trainY'],
    mnist_data['valX'],   mnist_data['valY'],
    mnist_data['testX'],  mnist_data['testY'],
    B_tot_list=B_tot_list
)

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
print("\nTask 3.3: Finding minimal bits for target accuracy...")
ta = MyTargetAllocator(K=3)
alphas = [0.6, 0.7, 0.8, 0.9, 0.95]
B_grid = [100, 400, 784, 1568, 2352, 3136]

cent_min = []
decz_min = []

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
print("All remaining plots generated!")
print("=" * 60)


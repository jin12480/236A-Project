"""Test script for Task 1 - Baseline Classifier"""
from utils import prepare_mnist_data, prepare_synthetic_data
from MySolution import MyDecentralized
import numpy as np

print("=" * 60)
print("Testing Task 1: Baseline Multi-class Linear Classifier")
print("=" * 60)

# Test with synthetic data first (smaller, faster)
print("\n1. Testing with Synthetic Data (2D, 3 classes)...")
try:
    syn_data = prepare_synthetic_data()
    print(f"   [OK] Data loaded: Train={syn_data['trainX'].shape}, Test={syn_data['testX'].shape}")
    
    clf_syn = MyDecentralized(K=3)
    print("   Training classifier...")
    clf_syn.train(syn_data['trainX'], syn_data['trainY'])
    print("   [OK] Training completed")
    
    train_acc = clf_syn.evaluate(syn_data['trainX'], syn_data['trainY'])
    test_acc = clf_syn.evaluate(syn_data['testX'], syn_data['testY'])
    print(f"   [OK] Synthetic Data Results:")
    print(f"     - Train Accuracy: {train_acc:.4f}")
    print(f"     - Test Accuracy:  {test_acc:.4f}")
except Exception as e:
    print(f"   [ERROR] Error with synthetic data: {e}")
    import traceback
    traceback.print_exc()

# Test with MNIST data
print("\n2. Testing with Fashion-MNIST Data (784D, 3 classes)...")
try:
    mnist_data = prepare_mnist_data()
    print(f"   [OK] Data loaded: Train={mnist_data['trainX'].shape}, Test={mnist_data['testX'].shape}")
    print(f"   Label distribution - Train: {np.bincount(mnist_data['trainY'].astype(int))}")
    print(f"   Label distribution - Test:  {np.bincount(mnist_data['testY'].astype(int))}")
    
    clf_mnist = MyDecentralized(K=3)
    print("   Training classifier (this may take a moment)...")
    clf_mnist.train(mnist_data['trainX'], mnist_data['trainY'])
    print("   [OK] Training completed")
    
    train_acc = clf_mnist.evaluate(mnist_data['trainX'], mnist_data['trainY'])
    test_acc = clf_mnist.evaluate(mnist_data['testX'], mnist_data['testY'])
    print(f"   [OK] Fashion-MNIST Results:")
    print(f"     - Train Accuracy: {train_acc:.4f}")
    print(f"     - Test Accuracy:  {test_acc:.4f}")
    
    # Check if weights are learned
    if clf_mnist.W is not None:
        print(f"   [OK] Model parameters learned:")
        print(f"     - W shape: {clf_mnist.W.shape}")
        print(f"     - b shape: {clf_mnist.b.shape}")
        print(f"     - W norm: {np.linalg.norm(clf_mnist.W):.4f}")
    
except Exception as e:
    print(f"   [ERROR] Error with MNIST data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Task 1 Testing Complete!")
print("=" * 60)


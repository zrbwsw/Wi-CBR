#!/usr/bin/env python3
"""
Test script for F1Calculator to verify its correctness.
"""

import torch
import numpy as np
from f1_calculator import F1Calculator, calculate_batch_f1
from sklearn.metrics import f1_score

def test_f1_calculator():
    """Test F1Calculator with known data."""
    print("Testing F1Calculator...")
    
    # Create test data
    num_classes = 6
    batch_size = 32
    
    # Generate some test predictions and targets
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test case 1: Perfect predictions
    print("\n=== Test Case 1: Perfect Predictions ===")
    targets = torch.randint(0, num_classes, (batch_size,))
    predictions = torch.zeros(batch_size, num_classes)
    for i, target in enumerate(targets):
        predictions[i, target] = 1.0  # Perfect prediction
    
    calculator = F1Calculator(num_classes)
    calculator.update(predictions, targets)
    metrics = calculator.compute_f1_scores()
    
    print(f"Accuracy: {metrics['accuracy']:.4f} (Expected: 1.0000)")
    print(f"Macro F1: {metrics['macro_f1']:.4f} (Expected: 1.0000)")
    print(f"Micro F1: {metrics['micro_f1']:.4f} (Expected: 1.0000)")
    
    # Test case 2: Random predictions
    print("\n=== Test Case 2: Random Predictions ===")
    targets = torch.randint(0, num_classes, (batch_size,))
    predictions = torch.randn(batch_size, num_classes)
    
    calculator = F1Calculator(num_classes)
    calculator.update(predictions, targets)
    metrics = calculator.compute_f1_scores()
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    
    # Verify with sklearn
    pred_classes = torch.argmax(predictions, dim=1).numpy()
    targets_np = targets.numpy()
    sklearn_f1 = f1_score(targets_np, pred_classes, average='macro', zero_division=0)
    print(f"Sklearn Macro F1: {sklearn_f1:.4f} (Should match our result)")
    
    # Test case 3: Multiple batches
    print("\n=== Test Case 3: Multiple Batches ===")
    calculator = F1Calculator(num_classes)
    
    for batch in range(5):
        targets = torch.randint(0, num_classes, (batch_size,))
        predictions = torch.randn(batch_size, num_classes)
        calculator.update(predictions, targets)
    
    metrics = calculator.compute_f1_scores()
    print(f"Multi-batch Accuracy: {metrics['accuracy']:.4f}")
    print(f"Multi-batch Macro F1: {metrics['macro_f1']:.4f}")
    
    # Test per-class F1
    per_class_f1 = calculator.compute_per_class_f1()
    print(f"\nPer-class F1 scores:")
    for class_name, f1 in per_class_f1.items():
        print(f"  {class_name}: {f1:.4f}")
    
    # Test batch F1 function
    print("\n=== Test Case 4: Batch F1 Function ===")
    targets = torch.randint(0, num_classes, (batch_size,))
    predictions = torch.randn(batch_size, num_classes)
    
    batch_f1 = calculate_batch_f1(predictions, targets, num_classes)
    print(f"Batch F1: {batch_f1:.4f}")
    
    print("\n=== Test Case 5: Edge Cases ===")
    # Empty calculator
    empty_calculator = F1Calculator(num_classes)
    empty_metrics = empty_calculator.compute_f1_scores()
    print(f"Empty calculator Macro F1: {empty_metrics['macro_f1']:.4f} (Expected: 0.0000)")
    
    # Single class predictions
    single_class_targets = torch.zeros(batch_size, dtype=torch.long)  # All class 0
    single_class_preds = torch.zeros(batch_size, num_classes)
    single_class_preds[:, 0] = 1.0  # All predict class 0
    
    single_calculator = F1Calculator(num_classes)
    single_calculator.update(single_class_preds, single_class_targets)
    single_metrics = single_calculator.compute_f1_scores()
    print(f"Single class Accuracy: {single_metrics['accuracy']:.4f} (Expected: 1.0000)")
    print(f"Single class Macro F1: {single_metrics['macro_f1']:.4f}")
    
    print("\nAll tests completed!")

def test_detailed_output():
    """Test detailed output functionality."""
    print("\n" + "="*60)
    print("TESTING DETAILED OUTPUT")
    print("="*60)
    
    # Create some test data
    num_classes = 6
    batch_size = 100
    
    calculator = F1Calculator(num_classes, class_names=[
        'Walk', 'Run', 'Sit', 'Stand', 'Jump', 'Wave'
    ])
    
    # Generate multiple batches of test data
    torch.manual_seed(123)
    for _ in range(10):
        targets = torch.randint(0, num_classes, (batch_size,))
        predictions = torch.randn(batch_size, num_classes)
        calculator.update(predictions, targets)
    
    # Print detailed results
    calculator.print_detailed_results()
    
    # Test saving to file
    test_file = '/tmp/test_f1_results.txt'
    calculator.save_results_to_file(test_file)
    print(f"\nResults saved to {test_file}")

if __name__ == '__main__':
    test_f1_calculator()
    test_detailed_output()
    print("\nF1Calculator testing completed successfully!")
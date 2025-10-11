import torch
import numpy as np
from typing import Tuple, Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

class F1Calculator:
    """
    F1 Score calculator for multi-class classification tasks.
    Supports both Macro and Micro F1 calculations.
    """
    
    def __init__(self, num_classes: int = 6, class_names: List[str] = None):
        """
        Initialize F1 calculator.
        
        Args:
            num_classes: Number of classes in the classification task
            class_names: Optional list of class names for better reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update the calculator with new predictions and targets.
        
        Args:
            predictions: Model predictions (logits or probabilities) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Convert predictions to class indices if they are logits/probabilities
        if predictions.dim() > 1:
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            predicted_classes = predictions
        
        # Convert to numpy and store
        pred_np = predicted_classes.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        self.all_predictions.extend(pred_np.tolist())
        self.all_targets.extend(target_np.tolist())
    
    def compute_f1_scores(self) -> Dict[str, float]:
        """
        Compute various F1 scores and related metrics.
        
        Returns:
            Dictionary containing different F1 scores and metrics
        """
        if not self.all_predictions or not self.all_targets:
            return {
                'macro_f1': 0.0,
                'micro_f1': 0.0,
                'weighted_f1': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'accuracy': 0.0
            }
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Calculate different F1 scores
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate precision and recall
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        
        return {
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'accuracy': float(accuracy)
        }
    
    def compute_per_class_f1(self) -> Dict[str, float]:
        """
        Compute F1 score for each class individually.
        
        Returns:
            Dictionary with per-class F1 scores
        """
        if not self.all_predictions or not self.all_targets:
            return {class_name: 0.0 for class_name in self.class_names}
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Calculate per-class F1 scores
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        return {
            self.class_names[i]: float(per_class_f1[i]) 
            for i in range(min(len(per_class_f1), len(self.class_names)))
        }
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            String containing detailed classification report
        """
        if not self.all_predictions or not self.all_targets:
            return "No data available for classification report."
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0,
            digits=4
        )
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.all_predictions or not self.all_targets:
            return np.zeros((self.num_classes, self.num_classes))
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
    
    def print_detailed_results(self):
        """Print detailed F1 calculation results."""
        print("\n" + "="*60)
        print("F1 SCORE EVALUATION RESULTS")
        print("="*60)
        
        # Overall metrics
        overall_metrics = self.compute_f1_scores()
        print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"Macro F1: {overall_metrics['macro_f1']:.4f}")
        print(f"Micro F1: {overall_metrics['micro_f1']:.4f}")
        print(f"Weighted F1: {overall_metrics['weighted_f1']:.4f}")
        print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {overall_metrics['macro_recall']:.4f}")
        
        # Per-class F1 scores
        per_class_f1 = self.compute_per_class_f1()
        print(f"\nPer-Class F1 Scores:")
        for class_name, f1_score in per_class_f1.items():
            print(f"  {class_name}: {f1_score:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(self.get_classification_report())
        print("="*60)
    
    def save_results_to_file(self, filepath: str):
        """
        Save F1 calculation results to file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            f.write("F1 SCORE EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            
            # Overall metrics
            overall_metrics = self.compute_f1_scores()
            f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1: {overall_metrics['macro_f1']:.4f}\n")
            f.write(f"Micro F1: {overall_metrics['micro_f1']:.4f}\n")
            f.write(f"Weighted F1: {overall_metrics['weighted_f1']:.4f}\n")
            f.write(f"Macro Precision: {overall_metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall: {overall_metrics['macro_recall']:.4f}\n")
            
            # Per-class F1 scores
            per_class_f1 = self.compute_per_class_f1()
            f.write(f"\nPer-Class F1 Scores:\n")
            for class_name, f1_score in per_class_f1.items():
                f.write(f"  {class_name}: {f1_score:.4f}\n")
            
            # Detailed classification report
            f.write(f"\nDetailed Classification Report:\n")
            f.write(self.get_classification_report())
            f.write("\n" + "="*60 + "\n")


def calculate_batch_f1(predictions: torch.Tensor, targets: torch.Tensor, 
                      num_classes: int = 6) -> float:
    """
    Calculate Macro F1 score for a single batch.
    
    Args:
        predictions: Model predictions [batch_size, num_classes] or [batch_size]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes
        
    Returns:
        Macro F1 score for the batch
    """
    calculator = F1Calculator(num_classes)
    calculator.update(predictions, targets)
    metrics = calculator.compute_f1_scores()
    return metrics['macro_f1']
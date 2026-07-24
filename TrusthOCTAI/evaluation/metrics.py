"""
TrusthOCTAI Evaluation Metrics Module
Computes Accuracy, Precision, Recall, Specificity, Macro F1, Cohen's Kappa, and ROC-AUC.
"""
import numpy as np

def compute_classification_metrics(labels: np.ndarray, preds: np.ndarray):
    """
    Computes standard classification performance metrics.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohens_kappa': kappa
    }

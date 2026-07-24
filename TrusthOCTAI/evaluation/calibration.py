"""
TrusthOCTAI Calibration Module
Computes Expected Calibration Error (ECE) and Brier Score.
"""
import numpy as np

def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)

def calculate_brier_score(probabilities: np.ndarray, targets: np.ndarray, num_classes: int = 4) -> float:
    """
    Computes Multi-Class Brier Score.
    """
    targets_one_hot = np.eye(num_classes)[targets]
    brier = np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1))
    return float(brier)

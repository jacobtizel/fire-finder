import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score, accuracy_score, precision_score, recall_score, f1_score

# --- Configuration ---
GROUND_TRUTH_MASK_DIR = 'BoWFireDataset/dataset/mask/fire'  # Path to your ground truth masks
PREDICTED_MASK_DIR = 'results/predicted_masks'  # Path to the directory where you save predicted masks
# Or if you saved segmented images:
# ORIGINAL_IMAGE_DIR = 'BoWFireDataset/dataset/img/fire'
# RESULTS_DIR = 'results/segmented_images'

RESULTS_OUTPUT_DIR = 'results/evaluation'
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask: {mask_path}")
        return None
    return mask

def generate_binary_mask(mask):
    if mask is None:
        return None
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return (binary_mask > 0).astype(np.uint8)

def evaluate_segmentation(ground_truth, predicted):
    if ground_truth is None or predicted is None:
        return {'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'cm': np.zeros((2, 2))}

    ground_truth_flat = ground_truth.flatten()
    predicted_flat = predicted.flatten()

    if ground_truth_flat.shape != predicted_flat.shape:
        print("Warning: Ground truth and predicted masks have different shapes.")
        return {'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'cm': np.zeros((2, 2))}

    tn, fp, fn, tp = confusion_matrix(ground_truth_flat, predicted_flat).ravel()
    cm = np.array([[tn, fp], [fn, tp]])

    if np.sum(ground_truth_flat) == 0 and np.sum(predicted_flat) == 0:
        iou = 1.0
    else:
        iou = jaccard_score(ground_truth_flat, predicted_flat, zero_division=0)

    accuracy = accuracy_score(ground_truth_flat, predicted_flat)
    precision = precision_score(ground_truth_flat, predicted_flat, zero_division=0)
    recall = recall_score(ground_truth_flat, predicted_flat, zero_division=0)
    f1 = f1_score(ground_truth_flat, predicted_flat, zero_division=0)

    return {'iou': iou, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}

# --- Main Evaluation Script ---

if __name__ == "__main__":
    ground_truth_mask_files = [f for f in os.listdir(GROUND_TRUTH_MASK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    results = {
        'rgb': {'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'cm': []},
        'ycbcr': {'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'cm': []},
        'lab': {'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'cm': []},
        'hsv': {'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'cm': []},
        'gabor': {'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'cm': []}
    }

    if not ground_truth_mask_files:
        print(f"No ground truth masks found in {GROUND_TRUTH_MASK_DIR}")
        exit()

    print("Evaluating saved segmentation results...")
    for mask_file in ground_truth_mask_files:
        mask_base_name = os.path.splitext(mask_file)[0]
        ground_truth_path = os.path.join(GROUND_TRUTH_MASK_DIR, mask_file)
        ground_truth_mask = load_mask(ground_truth_path)
        if ground_truth_mask is None:
            continue
        ground_truth_binary = generate_binary_mask(ground_truth_mask)
        if ground_truth_binary is None:
            continue

        print(f"\nEvaluating for {mask_file}:")

        for method in results.keys():
            predicted_mask_filename = f"{mask_base_name}_{method}_mask.png" # Example naming
            predicted_mask_path = os.path.join(PREDICTED_MASK_DIR, predicted_mask_filename)

            if os.path.exists(predicted_mask_path):
                predicted_mask = load_mask(predicted_mask_path)
                if predicted_mask is not None:
                    predicted_binary = generate_binary_mask(predicted_mask)
                    if predicted_binary is not None:
                        eval_metrics = evaluate_segmentation(ground_truth_binary, predicted_binary)
                        for metric, value in eval_metrics.items():
                            results[method][metric].append(value)
                        print(f"  {method.upper()}: IoU={eval_metrics['iou']:.4f}, Accuracy={eval_metrics['accuracy']:.4f}, F1={eval_metrics['f1']:.4f}")
                    else:
                        print(f"  Warning: Could not generate binary mask for {method} - {mask_file}")
                        for metric in results[method].keys():
                            results[method][metric].append(0.0 if metric != 'cm' else np.zeros((2, 2)))
                else:
                    print(f"  Warning: Could not load predicted mask for {method} - {mask_file}")
                    for metric in results[method].keys():
                        results[method][metric].append(0.0 if metric != 'cm' else np.zeros((2, 2)))
            else:
                print(f"  Warning: Predicted mask not found for {method} - {mask_file}")
                for metric in results[method].keys():
                    results[method][metric].append(0.0 if metric != 'cm' else np.zeros((2, 2)))

    print("\n--- Overall Performance ---")
    for method, metrics in results.items():
        print(f"\n--- {method.upper()} ---")
        avg_iou = np.mean(metrics['iou']) if metrics['iou'] else 0.0
        avg_accuracy = np.mean(metrics['accuracy']) if metrics['accuracy'] else 0.0
        avg_precision = np.mean(metrics['precision']) if metrics['precision'] else 0.0
        avg_recall = np.mean(metrics['recall']) if metrics['recall'] else 0.0
        avg_f1 = np.mean(metrics['f1']) if metrics['f1'] else 0.0
        avg_cm = np.sum(metrics['cm'], axis=0) if metrics['cm'] else np.zeros((2, 2))

        print(f"  Average IoU: {avg_iou:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1-Score: {avg_f1:.4f}")
        print(f"  Average Confusion Matrix:\n{avg_cm}")

    print(f"\nEvaluation results saved in '{RESULTS_OUTPUT_DIR}'.")
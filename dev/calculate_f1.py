import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd
import numpy as np

from main import FinancialDataReconciler, PlanDecisionEngine, DATASET_DIR

reconciler = FinancialDataReconciler(DATASET_DIR)
engine = PlanDecisionEngine(reconciler)
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

y_true_status = []
y_pred_status = []
y_true_method = []
y_pred_method = []

for idx, row in samples.iterrows():
    pred = engine.evaluate_request(row)
    y_true_status.append(row['affordability_status'])
    y_pred_status.append(pred['affordability_status'])
    y_true_method.append(row['recommended_payment_method'])
    y_pred_method.append(pred['recommended_payment_method'])

def compute_f1_metrics(y_true, y_pred, label_name):
    classes = sorted(list(set(y_true + y_pred)))
    metrics_per_class = {}
    
    total = len(y_true)
    macro_p, macro_r, macro_f1 = 0, 0, 0
    weighted_f1 = 0
    correct = 0

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics_per_class[c] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': support,
            'tp': tp, 'fp': fp, 'fn': fn
        }
        
        macro_p += prec
        macro_r += rec
        macro_f1 += f1
        weighted_f1 += f1 * (support / total)
        correct += tp

    num_classes = len(classes)
    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes
    micro_f1 = correct / total  # For multi-class single-label, micro-F1 == accuracy

    print(f"==================================================")
    print(f" F1 SCORE ANALYSIS: {label_name.upper()}")
    print(f"==================================================")
    print(f"{'Class':<24} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 72)
    for c, m in metrics_per_class.items():
        print(f"{c:<24} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1']*100:>8.2f}% | {m['support']:>8}")
    print("-" * 72)
    print(f"{'Micro Average / Accuracy':<24} | {'-':<10} | {'-':<10} | {micro_f1*100:>8.2f}% | {total:>8}")
    print(f"{'Macro Average':<24} | {macro_p*100:>8.2f}% | {macro_r*100:>8.2f}% | {macro_f1*100:>8.2f}% | {total:>8}")
    print(f"{'Weighted Average':<24} | {'-':<10} | {'-':<10} | {weighted_f1*100:>8.2f}% | {total:>8}")
    print()

compute_f1_metrics(y_true_method, y_pred_method, "Recommended Payment Method")
compute_f1_metrics(y_true_status, y_pred_status, "Affordability Status")


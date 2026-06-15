import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_metrics(preds: list, labels: list, class_names: list) -> dict:
    preds  = np.array(preds)
    labels = np.array(labels)

    acc             = float((preds == labels).mean())
    balanced_acc    = float(balanced_accuracy_score(labels, preds))
    macro_f1        = float(f1_score(labels, preds, average="macro",     zero_division=0))
    weighted_f1     = float(f1_score(labels, preds, average="weighted",  zero_division=0))
    macro_precision = float(precision_score(labels, preds, average="macro",    zero_division=0))
    macro_recall    = float(recall_score(labels, preds,    average="macro",    zero_division=0))
    kappa           = float(cohen_kappa_score(labels, preds))
    mcc             = float(matthews_corrcoef(labels, preds))

    per_class_f1        = f1_score(labels, preds,        average=None, zero_division=0)
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall    = recall_score(labels, preds,    average=None, zero_division=0)

    cm     = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0
    )

    return {
        "accuracy":           acc,
        "balanced_accuracy":  balanced_acc,
        "macro_f1":           macro_f1,
        "weighted_f1":        weighted_f1,
        "macro_precision":    macro_precision,
        "macro_recall":       macro_recall,
        "cohen_kappa":        kappa,
        "mcc":                mcc,
        "per_class_f1":       {c: float(f) for c, f in zip(class_names, per_class_f1)},
        "per_class_precision":{c: float(f) for c, f in zip(class_names, per_class_precision)},
        "per_class_recall":   {c: float(f) for c, f in zip(class_names, per_class_recall)},
        "confusion_matrix":   cm.tolist(),
        "report":             report,
    }

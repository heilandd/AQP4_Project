import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def evaluate_classifier_and_regressor(model, data_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    pred_labels = []
    true_labels = []
    logits = []
    region_pred = []
    region_true = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            data.edge_attr = data.distance.view(-1, 1).float()
            _, cls_out, pred_reg, _, _ = model(data)

            logits.append(cls_out.cpu().numpy())
            pred_labels.append(torch.argmax(cls_out, dim=1).cpu().numpy())
            true_labels.append(data.status.cpu().numpy())
            region_pred.append(pred_reg.cpu().numpy())
            region_true.append(data.region.cpu().numpy())

    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    logits = np.concatenate(logits)
    region_pred = np.concatenate(region_pred)
    region_true = np.concatenate(region_true)

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average="macro"),
        "recall": recall_score(true_labels, pred_labels, average="macro"),
        "f1": f1_score(true_labels, pred_labels, average="macro"),
        "confusion": confusion_matrix(true_labels, pred_labels),
    }
    print(metrics)

    fpr, tpr, _ = roc_curve(true_labels, logits[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.3f}")

    return {
        "pred_labels": pred_labels,
        "true_labels": true_labels,
        "logits": logits,
        "region_pred": region_pred,
        "region_true": region_true,
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }

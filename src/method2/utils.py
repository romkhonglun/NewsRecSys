import torch
import torchmetrics
import torch.nn.functional as F
import numpy as np


class AUC(torchmetrics.MeanMetric):
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        # Lọc bỏ padding (-1)
        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            return

            # Tính AUC cho từng sample trong batch rồi mean
        # Lưu ý: binary_auroc cần input phẳng hoặc batch chuẩn.
        # Ở đây ta loop qua từng user trong batch
        auc_scores = []
        for _p, _l in zip(preds, labels):
            _mask = _l != -1
            if _mask.sum() > 0:
                # Chỉ tính nếu có ít nhất 1 dương và 1 âm để tránh lỗi AUC
                if _l[_mask].sum() > 0 and (_l[_mask] == 0).sum() > 0:
                    score = torchmetrics.functional.classification.binary_auroc(
                        preds=_p[_mask], target=_l[_mask].long()
                    )
                    auc_scores.append(score)

        if len(auc_scores) > 0:
            value = torch.stack(auc_scores).mean()
            super().update(value=value, weight=len(auc_scores))


class NDCG5(torchmetrics.MeanMetric):
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        ndcg_scores = []
        for _p, _l in zip(preds, labels):
            _mask = _l != -1
            if _mask.sum() > 0:
                score = torchmetrics.functional.retrieval.retrieval_normalized_dcg(
                    preds=_p[_mask], target=_l[_mask].long(), top_k=5
                )
                ndcg_scores.append(score)

        if len(ndcg_scores) > 0:
            value = torch.stack(ndcg_scores).mean()
            super().update(value=value, weight=len(ndcg_scores))


class NDCG10(torchmetrics.MeanMetric):
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        ndcg_scores = []
        for _p, _l in zip(preds, labels):
            _mask = _l != -1
            if _mask.sum() > 0:
                score = torchmetrics.functional.retrieval.retrieval_normalized_dcg(
                    preds=_p[_mask], target=_l[_mask].long(), top_k=10
                )
                ndcg_scores.append(score)

        if len(ndcg_scores) > 0:
            value = torch.stack(ndcg_scores).mean()
            super().update(value=value, weight=len(ndcg_scores))


# Trong utils.py

def binary_listnet_loss(y_pred, y_true, eps=1e-5, padded_value_indicator=-1):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    # Masking bằng giá trị rất nhỏ để log_softmax hiểu là xác suất = 0
    y_pred[mask] = -1e9
    y_true[mask] = 0.0

    # Normalize y_true
    normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
    normalizer[normalizer == 0.0] = 1.0
    normalizer = normalizer.expand(-1, y_true.shape[1])
    y_true = torch.div(y_true, normalizer)

    # [FIX QUAN TRỌNG] Dùng log_softmax thay vì log(softmax)
    # Hàm này cực kỳ ổn định về số học
    preds_log = F.log_softmax(y_pred, dim=1)

    return torch.mean(-torch.sum(y_true * preds_log, dim=1))


class MetricsMeter(torch.nn.Module):
    def __init__(self, loss_weights: dict = None):
        super().__init__()
        # Mặc định weight nếu không truyền vào
        if loss_weights is None:
            self.loss_weights = {"bce_loss": 1.0, "listnet_loss": 0.5}
        else:
            self.loss_weights = loss_weights

        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.eval_metrics = torchmetrics.MetricCollection({
            "auc": AUC(),
            "ndcg@5": NDCG5(),
            "ndcg@10": NDCG10(),
        })
        self.reset()

    def reset(self):
        self.eval_metrics.reset()

    def update(self, batch: dict, n_samples: int = None):
        metrics = {}

        preds = batch["preds"]
        labels = batch["labels"]

        # Tạo mask loại bỏ padding (-1) hoặc các phần tử thừa
        # Giả sử padding label là -1 (như đã set trong dataset)
        # Hoặc dùng labels >= 0
        mask = labels != -1

        # 1. Tính BCE Loss (Pointwise)
        if "bce_loss" in self.loss_weights:
            # BCE cần labels float 0.0/1.0
            # Mask fill padding bằng 0 để không ảnh hưởng sum
            bce = self.bce_loss(preds, labels.float())
            bce = bce.masked_fill(~mask, 0)
            metrics["bce_loss"] = bce.sum() / (mask.sum() + 1e-6)

        # 3. Tính ListNet Loss (Listwise)
        if "listnet_loss" in self.loss_weights:
            metrics["listnet_loss"] = binary_listnet_loss(
                y_pred=preds,
                y_true=labels.float(),
                padded_value_indicator=-1
            )

        # Tổng hợp Loss
        metrics["loss"] = sum(w * metrics.get(k, 0.0) for k, w in self.loss_weights.items())

        # Cập nhật Metrics đánh giá (AUC, NDCG)
        # Chỉ lấy n_samples đầu nếu cần (cho validation nhanh)
        if n_samples:
            _preds = preds[:n_samples]
            _labels = labels[:n_samples]
        else:
            _preds, _labels = preds, labels

        self.eval_metrics.update(preds=_preds, labels=_labels)

        return metrics

    def compute(self, suffix: str = ""):
        # Trả về kết quả metrics (AUC, NDCG) đã tích lũy
        return {f"{k}{suffix}": v for k, v in self.eval_metrics.compute().items()}
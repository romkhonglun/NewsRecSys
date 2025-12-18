import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import numpy as np
from pathlib import Path

# Import Models
from model import TIME_FEATURE_NAML, TIME_FEATURE_NAMLConfig
# Import Utils vừa tạo
from utils import MetricsMeter


class NAMLLightningModule(L.LightningModule):
    def __init__(self, config=None, embedding_path=None, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])

        self.config = config if config is not None else TIME_FEATURE_NAMLConfig()

        # --- Init Model & Embeddings ---
        if embedding_path and Path(embedding_path).exists():
            print(f"Loading vectors from {embedding_path}...")
            vectors_np = np.load(embedding_path)
            vectors_tensor = torch.from_numpy(vectors_np).float()

            # Cập nhật config dim nếu khác
            real_dim = vectors_tensor.shape[1]
            if self.config.pretrained_dim != real_dim:
                self.config.pretrained_dim = real_dim

            norm = torch.norm(vectors_tensor, p=2, dim=1, keepdim=True)
            vectors_tensor = vectors_tensor / (norm + 1e-10)
        else:
            print("⚠️ Using Random Embeddings.")
            vectors_tensor = torch.randn(100000, self.config.pretrained_dim)

        self.model = TIME_FEATURE_NAML(self.config, vectors_tensor)

        # --- Metrics Meter (Thay cho criterion lẻ) ---
        # Kết hợp: 1.0 * BCE + 0.5 * ListNet
        # ListNet giúp học Ranking tốt hơn BCE thuần
        self.loss_weights = {"bce_loss": 1.0}
        self.meter = MetricsMeter(self.loss_weights)

    def forward(self, batch):
        return self.model(batch)

    # --- Hook Reset Metrics đầu mỗi Epoch ---
    def on_train_epoch_start(self):
        self.meter.reset()

    def on_validation_epoch_start(self):
        self.meter.reset()

    # --- Training Step ---
    def training_step(self, batch, batch_idx):
        # 1. Forward Pass
        # output trả về dict {'preds': ..., 'labels': ...} từ model
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                if torch.isnan(v).any():
                    raise ValueError(f"❌ Input '{k}' chứa NaN! Kiểm tra lại preprocess.")
                if torch.isinf(v).any():
                    raise ValueError(f"❌ Input '{k}' chứa Inf! Kiểm tra lại preprocess.")
        # ------------------------------

        output = self(batch)

        # 2. Update Meter
        # Trộn batch gốc (chứa labels gốc) với output (chứa preds)
        # Lưu ý: batch['labels'] có thể chứa padding (-1), output['labels'] cũng vậy
        # Ta cần đảm bảo pass đúng labels vào meter.

        # Gom lại thành dict cho meter
        meter_input = {
            "preds": output["preds"],  # [Batch, Num_Cand]
            "labels": batch["labels"]  # [Batch, Num_Cand] (chứa 1, 0, và pad)
        }

        # Hàm update trả về dict các loại loss
        losses = self.meter.update(meter_input)

        # 3. Logging
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['hist_indices'])
        )

        return losses["loss"]

    # --- Validation Step ---
    def validation_step(self, batch, batch_idx):
        output = self(batch)

        meter_input = {
            "preds": output["preds"],
            "labels": batch["labels"]
        }

        # Update metrics nhưng không cần lấy loss ngay (sẽ log ở epoch end)
        self.meter.update(meter_input)

    # --- Validation Epoch End ---
    def on_validation_epoch_end(self):
        # Tính toán metric tổng hợp (AUC, NDCG)
        metrics = self.meter.compute()

        # Log ra console/wandb
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_epoch=True, prog_bar=True
        )

        # In ra màn hình để dễ theo dõi
        print(f"\nEpoch {self.current_epoch} Val Metrics: {metrics}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        # Cosine Decay
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.trainer.estimated_stepping_batches if hasattr(self.trainer, 'estimated_stepping_batches') else 10000
        # )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            total_steps=10000,
            pct_start=0.1,
            anneal_strategy="cos"
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=2,
        #     min_lr=1e-6
        # )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
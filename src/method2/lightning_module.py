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
    def __init__(
        self,
        config=None,
        embedding_path=None,
        lr=1e-4,
        weight_decay=1e-5,
        scheduler="onecycle",
        scheduler_total_steps=None,
        scheduler_max_lr=None,
        scheduler_t_max=None,
    ):
        super().__init__()
        # save hyperparameters (will include scheduler and its params)
        self.save_hyperparameters(ignore=['config'])

        self.config = config if config is not None else TIME_FEATURE_NAMLConfig()

        # --- Init Model & Embeddings ---
        if embedding_path and Path(embedding_path).exists():
            print(f"Loading vectors from {embedding_path}...")
            vectors_np = np.load(embedding_path)
            vectors_tensor = torch.from_numpy(vectors_np).float()

            real_dim = vectors_tensor.shape[1]
            if self.config.pretrained_dim != real_dim:
                self.config.pretrained_dim = real_dim

            norm = torch.norm(vectors_tensor, p=2, dim=1, keepdim=True)
            vectors_tensor = vectors_tensor / (norm + 1e-10)
        else:
            print("⚠️ Using Random Embeddings.")
            vectors_tensor = torch.randn(100000, self.config.pretrained_dim)

        self.model = TIME_FEATURE_NAML(self.config, vectors_tensor)

        # --- Metrics Meter ---
        self.loss_weights = {"bce_loss": 1.0}
        self.meter = MetricsMeter(self.loss_weights)

    def forward(self, batch):
        return self.model(batch)

    def on_train_epoch_start(self):
        self.meter.reset()

    def on_validation_epoch_start(self):
        self.meter.reset()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            print("--- Đã nhận được batch đầu tiên trên TPU! ---")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                if torch.isnan(v).any():
                    raise ValueError(f"❌ Input '{k}' chứa NaN! Kiểm tra lại preprocess.")
                if torch.isinf(v).any():
                    raise ValueError(f"❌ Input '{k}' chứa Inf! Kiểm tra lại preprocess.")

        output = self(batch)
        meter_input = {"preds": output["preds"], "labels": batch["labels"]}
        losses = self.meter.update(meter_input)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['hist_indices'])
        )

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        meter_input = {"preds": output["preds"], "labels": batch["labels"]}
        self.meter.update(meter_input)

    def on_validation_epoch_end(self):
        metrics = self.meter.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_epoch=True, prog_bar=True
        )
        print(f"\nEpoch {self.current_epoch} Val Metrics: {metrics}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler_choice = getattr(self.hparams, "scheduler", "onecycle")

        # OneCycleLR (step-based)
        if scheduler_choice == "onecycle":
            # determine total_steps: explicit > trainer estimate > fallback
            total_steps = getattr(self.hparams, "scheduler_total_steps", None)

            if total_steps is None or total_steps <= 0:
                total_steps = 10000
            print(f"total_steps for OneCycleLR: {total_steps}")
            max_lr = getattr(self.hparams, "scheduler_max_lr", 3e-3)

            if max_lr is None:
                max_lr = 3e-3

            print(f"total_steps for OneCycleLR: {total_steps}, max_lr: {max_lr}")
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos"
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        # Cosine annealing (step-based)
        if scheduler_choice == "cosine":
            t_max = getattr(self.hparams, "scheduler_t_max", None)
            if t_max is None or t_max <= 0:
                t_max = 10000
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer
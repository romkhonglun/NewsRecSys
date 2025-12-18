import os
import argparse
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, TQDMProgressBar, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv

# Import các module đã tạo ở các bước trước
from dataset import NAMLDataModule
from model import TIME_FEATURE_NAMLConfig
from lightning_module import NAMLLightningModule

# Load biến môi trường (WANDB_API_KEY, etc.)
load_dotenv()

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG)
# ==========================================
# 1. Nơi chứa data đã chạy qua preprocess.py (quan trọng nhất)
# Lưu ý: preprocess.py lưu vào /kaggle/working/processed
PROCESSED_DIR = "/processed_parquet"

# 2. Đường dẫn đến file vector embedding (.npy)
EMBEDDING_PATH = "/embedding/body_emb.npy"


def parse_args():
    parser = argparse.ArgumentParser(description="Train NAML model with configurable paths and scheduler")
    parser.add_argument("--processed-dir", type=str, default=PROCESSED_DIR, help="Path to processed data directory")
    parser.add_argument("--embedding-path", type=str, default=EMBEDDING_PATH, help="Path to embedding .npy file")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["onecycle", "cosine", "reduceonplateau", "none"],
        default="onecycle",
        help="LR scheduler to use (default: onecycle)"
    )
    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        default=None,
        help="Total steps used by OneCycleLR (fallbacks to trainer estimate or 10000)"
    )
    parser.add_argument(
        "--scheduler-max-lr",
        type=float,
        default=None,
        help="Max LR for OneCycleLR (default inside module if not set)"
    )
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=None,
        help="T_max for CosineAnnealingLR (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max LR for OneCycleLR (default inside module if not set)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Max LR for OneCycleLR (default inside module if not set)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    L.seed_everything(42)  # Set seed để tái lập kết quả

    # 1. Init Config
    print("Initializing Configuration...")
    config = TIME_FEATURE_NAMLConfig()

    # In thông số kiểm tra
    print(f"Model Config: Window={config.window_size}, Interests={config.num_interests}")
    print(f"Using processed dir: {args.processed_dir}")
    print(f"Using embedding path: {args.embedding_path}")

    # 2. Init DataModule
    dm = NAMLDataModule(
        processed_dir=args.processed_dir,
        embedding_path=args.embedding_path,
        batch_size=args.batch_size,
        use_iterable=False,
        num_workers=2
    )

    # 3. Init Model (Lightning Module)
    model = NAMLLightningModule(
        config=config,
        embedding_path=args.embedding_path,
        lr=1e-3,
        weight_decay=1e-5,
        scheduler=args.lr_scheduler,
        scheduler_total_steps=args.scheduler_total_steps,
        scheduler_max_lr=args.scheduler_max_lr,
        scheduler_t_max=args.scheduler_t_max
    )

    # 4. Logger (Wandb)
    wandb_logger = WandbLogger(
        project="NAML-News-Rec",
        name="Variant-NAML-Final",
        log_model=False
    )

    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="naml-{epoch:02d}-{val/auc:.4f}",
        save_top_k=20,
        monitor="val/auc",
        mode="max",
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val/auc",
        min_delta=0.0001,
        patience=5,
        verbose=True,
        mode="max"
    )

    # 6. Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            ModelSummary(max_depth=2),
            TQDMProgressBar(refresh_rate=10)
        ],
        gradient_clip_algorithm="norm",
        max_epochs=args.epochs,
        precision="16-true",
        num_sanity_val_steps=0
    )

    print("🚀 Starting training...")
    trainer.fit(model, datamodule=dm)

    print(f"✅ Training finished. Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
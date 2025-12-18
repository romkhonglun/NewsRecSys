%%writefile
train.py
import os
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv

# Import các module đã tạo ở các bước trước
from dataset import NAMLDataModule
from variant_naml import VariantNAMLConfig
from time_feature_model import TIME_FEATURE_NAMLConfig
from lightning_module import NAMLLightningModule

# Load biến môi trường (WANDB_API_KEY, etc.)
load_dotenv()

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG)
# ==========================================
# 1. Nơi chứa data đã chạy qua preprocess.py (quan trọng nhất)
# Lưu ý: preprocess.py lưu vào /kaggle/working/processed
PROCESSED_DIR = "/kaggle/working/processed"

# 2. Đường dẫn đến file vector embedding (.npy)
# Nếu bạn chưa có file này, hãy tạo dummy hoặc trỏ tạm vào đâu đó.
# Model sẽ tự tạo random nếu không tìm thấy file này (như logic trong lightning_module.py)
EMBEDDING_PATH = "/kaggle/working/processed_data/body_emb.npy"


def main():
    L.seed_everything(42)  # Set seed để tái lập kết quả

    # 1. Init Config
    print("Initializing Configuration...")
    config = TIME_FEATURE_NAMLConfig()

    # In thông số kiểm tra
    print(f"Model Config: Window={config.window_size}, Interests={config.num_interests}")

    # 2. Init DataModule
    # Lưu ý: Class này giờ nhận 'processed_dir' chứ không phải 'root_path'
    dm = NAMLDataModule(
        processed_dir=PROCESSED_DIR,
        embedding_path=EMBEDDING_PATH,
        batch_size=512,  # Tăng lên nếu VRAM còn trống (512 là an toàn cho T4 x2)
        history_len=30,
        num_workers=2  # Kaggle có 2 core CPU mạnh hoặc 4 core yếu, để 2 là an toàn
    )

    # 3. Init Model (Lightning Module)
    # Lưu ý: Bỏ tham số 'mode' vì code mới chỉ chạy VariantNAML
    model = NAMLLightningModule(
        config=config,
        embedding_path=EMBEDDING_PATH,
        lr=1e-3,
        weight_decay=1e-5
    )

    # 4. Logger (Wandb)
    # Set log_model=False để đỡ tốn dung lượng upload model lên mây
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
        patience=5,  # Giảm patience xuống 3 để tiết kiệm thời gian GPU Kaggle
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
        max_epochs=20,  # Train nhiều epoch hơn (Early Stop sẽ lo phần dừng)
        # log_every_n_steps=50,

        # [QUAN TRỌNG] Precision 16-mixed giúp giảm 1/2 VRAM và train nhanh gấp đôi trên T4
        precision="32",

        # Cắt gradient để ổn định Transformer training
        # gradient_clip_val=0.5,

        # Kiểm tra validation loop trước khi train để đảm bảo code không bug
        # num_sanity_val_steps=2
    )

    print("🚀 Starting training...")
    trainer.fit(model, datamodule=dm)

    print(f"✅ Training finished. Best model path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
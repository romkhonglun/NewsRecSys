import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from pathlib import Path
import pytorch_lightning as L
import torch.distributed as dist


# ==========================================
# 1. QUẢN LÝ STATIC ARTICLE FEATURES
# ==========================================
class StaticArticleFeatures:
    def __init__(self, processed_path):
        print(f"📖 Loading Article Features: {processed_path}")
        df = pl.read_parquet(processed_path)
        max_id = df["id_encoded"].max() or 0
        vocab_size = int(max_id) + 1

        self.num_mat = np.zeros((vocab_size, 5), dtype=np.float32)
        self.cat_mat = np.zeros((vocab_size, 1), dtype=np.int32)

        ids = df["id_encoded"].to_numpy().astype(np.int32)
        cols_num = ["norm_views", "norm_inviews", "sentiment_score", "norm_read_time", "published_time"]

        vals_num = df.select(cols_num).to_numpy().astype(np.float32)
        self.num_mat[ids] = np.nan_to_num(vals_num, nan=0.0)

        vals_cat = df.select("cat_encoded").to_numpy().astype(np.int32)
        self.cat_mat[ids] = vals_cat

    def get(self, indices):
        return self.num_mat[indices], self.cat_mat[indices]


# ==========================================
# 2. BASE LOGIC (Xử lý chung)
# ==========================================
class NewsBaseLogic:
    def _init_base(self, history_path, article_features, history_len, neg_ratio):
        self.art_feats = article_features
        self.history_len = history_len
        self.neg_ratio = neg_ratio

        print(f"📦 Pre-loading History: {history_path}")
        df_hist = pl.read_parquet(history_path)
        if df_hist["user_id"].dtype == pl.Utf8:
            df_hist = df_hist.with_columns(pl.col("user_id").cast(pl.Int32))

        max_uid = df_hist["user_id"].max() or 0
        num_users = int(max_uid) + 1

        self.hist_ids_mat = np.zeros((num_users, history_len), dtype=np.int32)
        self.hist_ts_mat = np.zeros((num_users, history_len), dtype=np.float64)
        self.hist_lens = np.zeros(num_users, dtype=np.int32)

        for row in df_hist.iter_rows(named=True):
            uid = row["user_id"]
            ids = row["hist_ids"][-history_len:] if row["hist_ids"] else []
            l = len(ids)
            if l > 0:
                self.hist_ids_mat[uid, :l] = ids
                self.hist_ts_mat[uid, :l] = np.array(row["hist_ts"][:l], dtype=np.float64)
                self.hist_lens[uid] = l

    def _process_row(self, row):
        uid = int(row["user_id"])
        imp_ts = float(row["imp_ts"])

        h_ids = self.hist_ids_mat[uid]
        h_ts = self.hist_ts_mat[uid]
        curr_len = self.hist_lens[uid]

        ts_diff = np.zeros(self.history_len, dtype=np.float32)
        if curr_len > 0:
            ts_diff[:curr_len] = np.log1p(np.clip((imp_ts - h_ts[:curr_len]) / 3600.0, 0, None))

        inv, clk = row["inv_ids"], row["clk_ids"]
        pos_id = np.random.choice(clk) if clk else (inv[0] if inv else 0)
        neg_pool = list(set(inv) - set(clk))

        if len(neg_pool) >= self.neg_ratio:
            neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False)
        else:
            neg_ids = [neg_pool[i % len(neg_pool)] if neg_pool else pos_id for i in range(self.neg_ratio)]

        cand_ids = np.array([pos_id] + list(neg_ids), dtype=np.int32)
        cand_nums, cand_cats = self.art_feats.get(cand_ids)
        cand_nums[:, 4] = np.log1p(np.abs(imp_ts - np.nan_to_num(cand_nums[:, 4], nan=imp_ts)) / 3600.0)

        return {
            "hist_indices": torch.LongTensor(h_ids.astype(np.int64)),
            "hist_diff": torch.FloatTensor(ts_diff),
            "cand_indices": torch.LongTensor(cand_ids.astype(np.int64)),
            "cand_num": torch.FloatTensor(cand_nums),
            "cand_cat": torch.LongTensor(cand_cats.flatten().astype(np.int64)),
            "imp_feats": torch.FloatTensor(
                [np.log1p(curr_len), (imp_ts % 86400) / 86400.0, float(row["norm_age"] or 0)]),
            "labels": torch.FloatTensor([1.0] + [0.0] * self.neg_ratio)
        }


# ==========================================
# OPTION 1: MAP-STYLE (Dùng khi đủ RAM nạp Behaviors)
# ==========================================
class NewsMapDataset(Dataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, history_len=30, neg_ratio=4):
        self._init_base(history_path, article_features, history_len, neg_ratio)
        df = pl.read_parquet(behaviors_path)
        self.behaviors = df.to_dicts()  # Hoặc dùng mảng như code trước nếu RAM ít

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        return self._process_row(self.behaviors[idx])


# ==========================================
# OPTION 2: ITERABLE (Dùng cho dữ liệu khổng lồ)
# ==========================================
class NewsStreamDataset(IterableDataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, history_len=30, neg_ratio=4, batch_size=32):
        self._init_base(history_path, article_features, history_len, neg_ratio)
        self.behaviors_path = behaviors_path
        self.batch_size = batch_size

    def _get_stream(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Sharding đơn giản cho TPU Pods/DDP
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        pf = pq.ParquetFile(self.behaviors_path)
        # Đọc theo batch để CPU không bị quá tải
        for batch in pf.iter_batches(batch_size=self.batch_size * 20):
            df_batch = pl.from_arrow(batch)
            for i, row in enumerate(df_batch.iter_rows(named=True)):
                if (i + worker_id + rank) % (num_workers * world_size) == 0:
                    yield row

    def __iter__(self):
        for item in self._get_stream():
            processed = self._process_row(item)
            if processed: yield processed


# ==========================================
# 3. DATAMODULE
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, use_iterable=False, batch_size=256, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        art_feats = StaticArticleFeatures(Path(self.hparams.processed_dir) / "articles_processed.parquet")

        DatasetClass = NewsStreamDataset if self.hparams.use_iterable else NewsMapDataset

        if stage in ('fit', None):
            self.train_ds = DatasetClass(
                Path(self.hparams.processed_dir) / "train/behaviors_processed.parquet",
                Path(self.hparams.processed_dir) / "train/history_processed.parquet",
                art_feats, batch_size=self.hparams.batch_size
            )
            self.val_ds = DatasetClass(
                Path(self.hparams.processed_dir) / "validation/behaviors_processed.parquet",
                Path(self.hparams.processed_dir) / "validation/history_processed.parquet",
                art_feats, batch_size=self.hparams.batch_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size,
            shuffle=(not self.hparams.use_iterable),
            num_workers=self.hparams.num_workers,
            pin_memory=False, drop_last=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=False, persistent_workers=True
        )
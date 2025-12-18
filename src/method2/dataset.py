import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from pathlib import Path
import pytorch_lightning as L
import torch.distributed as dist
from itertools import islice


# ==========================================
# 1. QUẢN LÝ VECTOR EMBEDDING (Optimized)
# ==========================================
class NewsEmbeddingManager:
    def __init__(self, embedding_path):
        self.embedding_path = Path(embedding_path)
        print(f"🚀 Loading Article Vectors: {self.embedding_path}")

        try:
            # Load với mmap_mode nếu file cực lớn để tiết kiệm RAM
            self.vectors = np.load(self.embedding_path).astype(np.float32)

            # L2 Normalize sẵn để khi tính similarity chỉ cần dùng Dot Product
            norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = np.divide(self.vectors, norm, out=np.zeros_like(self.vectors), where=norm != 0)

            # Xử lý NaN sau khi normalize
            self.vectors = np.nan_to_num(self.vectors, nan=0.0)
            print(f"✅ Vectors Ready. Shape: {self.vectors.shape}")
        except Exception as e:
            print(f"⚠️ Warning: {e}. Using Random.")
            self.vectors = np.random.randn(100000, 768).astype(np.float32)

    def get_vectors_by_indices(self, indices):
        return self.vectors[indices]


# ==========================================
# 2. STATIC ARTICLE FEATURES (Numpy Matrix)
# ==========================================
class StaticArticleFeatures:
    def __init__(self, processed_path):
        df = pl.read_parquet(processed_path)
        max_id = df["id_encoded"].max() or 0
        vocab_size = max_id + 1

        self.num_mat = np.zeros((vocab_size, 5), dtype=np.float32)
        self.cat_mat = np.zeros((vocab_size, 1), dtype=np.int32)

        ids = df["id_encoded"].to_numpy()
        cols_num = ["norm_views", "norm_inviews", "sentiment_score", "norm_read_time", "published_time"]

        vals_num = df.select(cols_num).to_numpy().astype(np.float32)
        self.num_mat[ids] = np.nan_to_num(vals_num, nan=0.0)

        vals_cat = df.select("cat_encoded").to_numpy().astype(np.int32)
        self.cat_mat[ids] = vals_cat

    def get(self, indices):
        return self.num_mat[indices], self.cat_mat[indices]


class NewsBaseLogic:
    """Chứa các hàm xử lý logic để tránh lặp code"""

    def _init_base(self, history_path, article_features, embedding_manager, history_len, neg_ratio):
        self.art_feats = article_features
        self.emb_manager = embedding_manager
        self.history_len = history_len
        self.neg_ratio = neg_ratio
        self._load_history_to_numpy(history_path)

    def _load_history_to_numpy(self, path):
        print(f"📦 Pre-loading History from {path}...")
        df = pl.read_parquet(path)
        if df["user_id"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("user_id").cast(pl.Int32))

        max_uid = df["user_id"].max() or 0
        num_users = int(max_uid) + 1

        self.hist_ids_mat = np.zeros((num_users, self.history_len), dtype=np.int32)
        self.hist_scr_mat = np.zeros((num_users, self.history_len), dtype=np.float32)
        self.hist_tm_mat = np.zeros((num_users, self.history_len), dtype=np.float32)
        self.hist_ts_mat = np.zeros((num_users, self.history_len), dtype=np.float64)
        self.hist_lens = np.zeros(num_users, dtype=np.int32)

        for row in df.iter_rows(named=True):
            uid = row["user_id"]
            ids = row["hist_ids"][-self.history_len:] if row["hist_ids"] else []
            l = len(ids)
            if l > 0:
                self.hist_ids_mat[uid, :l] = ids
                clean = lambda x: np.nan_to_num(np.array(x[:l], dtype=np.float32), 0.0)
                self.hist_scr_mat[uid, :l] = clean(row["hist_scroll"])
                self.hist_tm_mat[uid, :l] = clean(row["hist_time"])
                self.hist_ts_mat[uid, :l] = np.array(row["hist_ts"][:l], dtype=np.float64)
                self.hist_lens[uid] = l

    def _process_row(self, row):
        # Logic xử lý row giữ nguyên như code cũ của bạn
        try:
            user_id = int(row["user_id"])
        except:
            return None

        imp_ts = row["imp_ts"] or 0.0

        # 1. Truy xuất history
        if user_id >= self.hist_ids_mat.shape[0]:
            h_ids, h_scr, h_tm, h_ts, curr_len = np.zeros(self.history_len, dtype=np.int32), np.zeros(
                self.history_len), np.zeros(self.history_len), np.zeros(self.history_len), 0
        else:
            h_ids, h_scr, h_tm, h_ts, curr_len = self.hist_ids_mat[user_id], self.hist_scr_mat[user_id], \
            self.hist_tm_mat[user_id], self.hist_ts_mat[user_id], self.hist_lens[user_id]

        # 2. Recency
        ts_diff_log = np.zeros(self.history_len, dtype=np.float32)
        if curr_len > 0:
            diffs = (imp_ts - h_ts[:curr_len]) / 3600.0
            ts_diff_log[:curr_len] = np.log1p(np.clip(diffs, 0, None))

        # 3. Sampling & Features
        inv_ids, clk_ids = row["inv_ids"] or [], row["clk_ids"] or []
        pos_id = np.random.choice(clk_ids) if clk_ids else (inv_ids[0] if inv_ids else 0)
        neg_pool = list(set(inv_ids) - set(clk_ids))

        if len(neg_pool) >= self.neg_ratio:
            neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False).tolist()
        else:
            neg_ids = [neg_pool[i % len(neg_pool)] if neg_pool else pos_id for i in range(self.neg_ratio)]

        candidate_ids = [pos_id] + neg_ids
        cand_nums, cand_cats = self.art_feats.get(candidate_ids)
        cand_nums[:, 4] = np.log1p(np.abs(imp_ts - np.nan_to_num(cand_nums[:, 4], nan=imp_ts)) / 3600.0)

        # 4. Sim
        cand_vecs = self.emb_manager.get_vectors_by_indices(candidate_ids)
        scores = (cand_vecs @ np.mean(self.emb_manager.get_vectors_by_indices(h_ids[:curr_len]), axis=0)).reshape(-1,
                                                                                                                  1) if curr_len > 0 else np.zeros(
            (len(candidate_ids), 1))

        return {
            "hist_indices": torch.from_numpy(h_ids.astype(np.int64)),
            "hist_scroll": torch.from_numpy(h_scr),
            "hist_time": torch.from_numpy(h_tm),
            "hist_diff": torch.from_numpy(ts_diff_log),
            "cand_indices": torch.tensor(candidate_ids, dtype=torch.long),
            "cand_num": torch.from_numpy(cand_nums),
            "cand_cat": torch.from_numpy(cand_cats).long(),
            "cand_sim": torch.from_numpy(np.nan_to_num(scores, 0.0)).float(),
            "imp_feats": torch.tensor([np.log1p(curr_len), (imp_ts % 86400) / 86400.0, float(row["norm_age"] or 0.0)],
                                      dtype=torch.float),
            "labels": torch.tensor([1.0] + [0.0] * self.neg_ratio, dtype=torch.float)
        }


# ==========================================
# OPTION 1: MAP-STYLE DATASET (Dataset thường)
# ==========================================
class NewsMapDataset(Dataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, embedding_manager,
                 history_len=30, neg_ratio=4, **kwargs):
        self._init_base(history_path, article_features, embedding_manager, history_len, neg_ratio)

        # Thay vì to_dicts(), hãy để nó ở dạng Polars DataFrame hoặc Numpy Array
        df = pl.read_parquet(behaviors_path)
        self.user_ids = df["user_id"].to_numpy()
        self.imp_ts = df["imp_ts"].to_numpy()
        self.inv_ids = df["inv_ids"].to_list()
        self.clk_ids = df["clk_ids"].to_list()
        self.norm_age = df["norm_age"].to_numpy()
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Tạo một dictionary giả lập 'row' để tận dụng logic _process_row hiện tại
        row = {
            "user_id": self.user_ids[idx],
            "imp_ts": self.imp_ts[idx],
            "inv_ids": self.inv_ids[idx],
            "clk_ids": self.clk_ids[idx],
            "norm_age": self.norm_age[idx]
        }
        return self._process_row(row)


# ==========================================
# OPTION 2: ITERABLE DATASET (Streaming)
# ==========================================
class NewsStreamDataset(IterableDataset, NewsBaseLogic):
    def __init__(self, behaviors_path, history_path, article_features, embedding_manager,
                 history_len=30, neg_ratio=4, batch_size=32, shuffle_buffer=1000, mode='train'):
        self._init_base(history_path, article_features, embedding_manager, history_len, neg_ratio)
        self.behaviors_path = behaviors_path
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.mode = mode

    def _get_stream(self):
        # Sharding cho multi-worker và DDP
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        pf = pq.ParquetFile(self.behaviors_path)
        # Bỏ qua dòng theo shard
        start_idx = rank * num_workers + worker_id
        step = world_size * num_workers

        for batch in pf.iter_batches(batch_size=self.batch_size * 10):
            batch_dict = batch.to_pydict()
            keys = list(batch_dict.keys())
            for i in range(start_idx, len(batch_dict[keys[0]]), step):
                yield {k: batch_dict[k][i] for k in keys}

    def __iter__(self):
        stream = self._get_stream()
        if self.mode == 'train':
            buffer = []
            for item in stream:
                processed = self._process_row(item)
                if processed: buffer.append(processed)
                if len(buffer) >= self.shuffle_buffer:
                    yield buffer.pop(np.random.randint(len(buffer)))
            while buffer: yield buffer.pop(np.random.randint(len(buffer)))
        else:
            for item in stream:
                processed = self._process_row(item)
                if processed: yield processed


# ==========================================
# DATA MODULE HỖ TRỢ CẢ 2
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, embedding_path, use_iterable=False, batch_size=64, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.use_iterable = use_iterable

    def setup(self, stage=None):
        art_feats = StaticArticleFeatures(Path(self.hparams.processed_dir) / "articles_processed.parquet")
        emb_manager = NewsEmbeddingManager(self.hparams.embedding_path)

        DatasetClass = NewsStreamDataset if self.use_iterable else NewsMapDataset

        if stage in ('fit', None):
            self.train_ds = DatasetClass(
                Path(self.hparams.processed_dir) / "train/behaviors_processed.parquet",
                Path(self.hparams.processed_dir) / "train/history_processed.parquet",
                art_feats, emb_manager, mode='train'
            )
            self.val_ds = DatasetClass(
                Path(self.hparams.processed_dir) / "validation/behaviors_processed.parquet",
                Path(self.hparams.processed_dir) / "validation/history_processed.parquet",
                art_feats, emb_manager, mode='val'
            )

    def train_dataloader(self):
        # Lưu ý: Với MapDataset có thể dùng shuffle=True, IterableDataset thì KHÔNG
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=(not self.use_iterable),
                          num_workers=self.hparams.num_workers,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,persistent_workers=True)
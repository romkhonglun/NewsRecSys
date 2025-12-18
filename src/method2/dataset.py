import torch
import polars as pl
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
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


# ==========================================
# 3. MAIN DATASET (Numpy History & Streaming)
# ==========================================
class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, behaviors_path, history_path, article_features, embedding_manager,
                 history_len=30, neg_ratio=4, batch_size=32, mode='train', shuffle_buffer=1000):
        super().__init__()
        self.behaviors_path = behaviors_path
        self.art_feats = article_features
        self.emb_manager = embedding_manager
        self.history_len = history_len
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer

        # Load History vào Ma trận Numpy thay vì Dictionary
        self._load_history_to_numpy(history_path)

    def _load_history_to_numpy(self, path):
        print(f"📦 Pre-loading History into Numpy Matrices...")
        # Thêm .with_columns để ép kiểu user_id sang int
        df = pl.read_parquet(path).with_columns(
            pl.col("user_id").cast(pl.Int32)
        )

        max_uid = df["user_id"].max() or 0
        num_users = int(max_uid) + 1

        # Khởi tạo ma trận rỗng (Pre-padded với 0)
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
                self.hist_scr_mat[uid, :l] = np.nan_to_num(row["hist_scroll"][:l], nan=0.0)
                self.hist_tm_mat[uid, :l] = np.nan_to_num(row["hist_time"][:l], nan=0.0)
                self.hist_ts_mat[uid, :l] = np.nan_to_num(row["hist_ts"][:l], nan=0.0)
                self.hist_lens[uid] = l

    def _process_row(self, row):
        user_id = row["user_id"]
        imp_ts = row["imp_ts"] or 0.0

        # --- 1. TRUY XUẤT LỊCH SỬ (O(1) complexity) ---
        h_ids = self.hist_ids_mat[user_id]
        h_scr = self.hist_scr_mat[user_id]
        h_tm = self.hist_tm_mat[user_id]
        h_ts = self.hist_ts_mat[user_id]
        curr_len = self.hist_lens[user_id]

        # --- 2. TÍNH RECENCY (Vectorized) ---
        ts_diff_log = np.zeros(self.history_len, dtype=np.float32)
        if curr_len > 0:
            # Chỉ tính cho các phần tử thực (không phải padding)
            diffs = (imp_ts - h_ts[:curr_len]) / 3600.0
            ts_diff_log[:curr_len] = np.log1p(np.clip(diffs, 0, None))

        # --- 3. NEGATIVE SAMPLING (Fast) ---
        inv_ids = row["inv_ids"] or []
        clk_ids = row["clk_ids"] or []

        pos_id = np.random.choice(clk_ids) if clk_ids else (inv_ids[0] if inv_ids else 0)
        neg_pool = list(set(inv_ids) - set(clk_ids))

        if not neg_pool:
            neg_ids = [pos_id] * self.neg_ratio
        elif len(neg_pool) >= self.neg_ratio:
            neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False).tolist()
        else:
            neg_ids = [neg_pool[i % len(neg_pool)] for i in range(self.neg_ratio)]

        candidate_ids = [pos_id] + neg_ids

        # --- 4. ARTICLE FEATURES & FRESHNESS ---
        cand_nums, cand_cats = self.art_feats.get(candidate_ids)
        # Dynamic Freshness: log(abs(now - pub_time))
        cand_nums[:, 4] = np.log1p(np.abs(imp_ts - np.nan_to_num(cand_nums[:, 4], nan=imp_ts)) / 3600.0)

        # --- 5. CANDIDATE SIMILARITY (Vectorized Dot Product) ---
        cand_vecs = self.emb_manager.get_vectors_by_indices(candidate_ids)
        if curr_len > 0:
            hist_vecs = self.emb_manager.get_vectors_by_indices(h_ids[:curr_len])
            user_vec = np.mean(hist_vecs, axis=0)
            scores = (cand_vecs @ user_vec).reshape(-1, 1)  # Matrix multiplication
        else:
            scores = np.zeros((len(candidate_ids), 1), dtype=np.float32)

        # --- 6. TENSOR CONVERSION ---
        return {
            "hist_indices": torch.from_numpy(h_ids.astype(np.int64)),
            "hist_scroll": torch.from_numpy(h_scr),
            "hist_time": torch.from_numpy(h_tm),
            "hist_diff": torch.from_numpy(ts_diff_log),
            "cand_indices": torch.tensor(candidate_ids, dtype=torch.long),
            "cand_num": torch.from_numpy(cand_nums),
            "cand_cat": torch.from_numpy(cand_cats).long(),
            "cand_sim": torch.from_numpy(np.nan_to_num(scores, 0.0)),
            "imp_feats": torch.tensor([np.log1p(curr_len), (imp_ts % 86400) / 86400.0, row["norm_age"] or 0.0],
                                      dtype=torch.float),
            "device_type": torch.tensor(row["device_type"] or 0, dtype=torch.long),
            "labels": torch.tensor([1.0] + [0.0] * self.neg_ratio, dtype=torch.float)
        }

    def _stream_parquet(self):
        pf = pq.ParquetFile(self.behaviors_path)
        # Tăng batch_size của Parquet để đọc từ disk nhanh hơn
        for batch in pf.iter_batches(batch_size=self.batch_size * 20):
            batch_dict = batch.to_pydict()
            keys = list(batch_dict.keys())
            length = len(batch_dict[keys[0]])
            for i in range(length):
                yield {k: batch_dict[k][i] for k in keys}

    def __iter__(self):
        # ... (Phần logic worker/dist sharding giữ nguyên như cũ) ...
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        iterator = islice(self._stream_parquet(), rank * num_workers + worker_id, None, world_size * num_workers)

        # Buffer-based Shuffling
        if self.mode == 'train':
            buffer = []
            for item in iterator:
                buffer.append(self._process_row(item))
                if len(buffer) >= self.shuffle_buffer:
                    idx = np.random.randint(len(buffer))
                    yield buffer.pop(idx)
            np.random.shuffle(buffer)
            yield from buffer
        else:
            for item in iterator:
                yield self._process_row(item)


# ==========================================
# 4. LIGHTNING DATA MODULE
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, embedding_path, batch_size=64, history_len=30, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.processed_dir = Path(processed_dir)
        self.embedding_path = embedding_path
        self.art_feats = None
        self.emb_manager = None

    def setup(self, stage=None):
        if self.art_feats is None:
            self.art_feats = StaticArticleFeatures(self.processed_dir / "articles_processed.parquet")
            self.emb_manager = NewsEmbeddingManager(self.embedding_path)

        common = {'article_features': self.art_feats, 'embedding_manager': self.emb_manager,
                  'history_len': self.hparams.history_len, 'batch_size': self.hparams.batch_size}

        if stage in ('fit', None):
            self.train_ds = PreprocessedIterableDataset(self.processed_dir / "train/behaviors_processed.parquet",
                                                        self.processed_dir / "train/history_processed.parquet",
                                                        mode='train', **common)
            self.val_ds = PreprocessedIterableDataset(self.processed_dir / "validation/behaviors_processed.parquet",
                                                      self.processed_dir / "validation/history_processed.parquet",
                                                      mode='val', **common)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=True, prefetch_factor=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=True, prefetch_factor=2, persistent_workers=True)
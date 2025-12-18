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
# 1. QUẢN LÝ VECTOR EMBEDDING
# ==========================================
class NewsEmbeddingManager:
    def __init__(self, embedding_path):
        self.embedding_path = Path(embedding_path)
        print(f"Loading Article Vectors from {self.embedding_path}...")

        try:
            self.vectors = np.load(self.embedding_path).astype(np.float32)
            # Check NaN trong vectors gốc
            if np.isnan(self.vectors).any():
                print("⚠️ Found NaN in embedding file! Replacing with 0.")
                self.vectors = np.nan_to_num(self.vectors, nan=0.0)

            # L2 Normalize
            norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norm[norm == 0] = 1e-10
            self.vectors = self.vectors / norm
            print(f"✅ Vectors loaded. Shape: {self.vectors.shape}")
        except FileNotFoundError:
            print(f"⚠️ Warning: Embedding file not found. Using Random.")
            self.vectors = np.random.randn(100000, 768).astype(np.float32)

    def get_vectors_by_indices(self, indices):
        return self.vectors[indices]


# ==========================================
# 2. STATIC ARTICLE FEATURES
# ==========================================
class StaticArticleFeatures:
    def __init__(self, processed_path):
        print(f"Loading Static Article Features from {processed_path}...")
        df = pl.read_parquet(processed_path)

        max_id = df["id_encoded"].max()
        if max_id is None: max_id = 0
        vocab_size = max_id + 1

        self.num_mat = np.zeros((vocab_size, 5), dtype=np.float32)
        self.cat_mat = np.zeros((vocab_size, 1), dtype=np.int32)

        ids = df["id_encoded"].to_numpy()

        cols_num = ["norm_views", "norm_inviews", "sentiment_score", "norm_read_time", "published_time"]
        vals_num = df.select(cols_num).to_numpy().astype(np.float32)

        # [FIX] Clean NaN trong Metadata
        vals_num = np.nan_to_num(vals_num, nan=0.0)

        self.num_mat[ids] = vals_num

        vals_cat = df.select("cat_encoded").to_numpy().astype(np.int32)
        self.cat_mat[ids] = vals_cat

        print(f"✅ Article Features Loaded. Vocab Size: {vocab_size}")

    def get(self, indices):
        return self.num_mat[indices], self.cat_mat[indices]


# ==========================================
# 3. MAIN DATASET (STREAMING)
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

        self.history_map = self._load_history_to_ram(history_path)

    def _load_history_to_ram(self, path):
        print(f"Loading History Map from {path}...")
        df = pl.read_parquet(path)

        data_map = {}
        for row in df.iter_rows(named=True):
            uid = row["user_id"]

            # [FIX] Đảm bảo dữ liệu load lên sạch NaN ngay từ đầu
            # Nếu hist_ids rỗng hoặc None, trả về array rỗng
            ids = row["hist_ids"] if row["hist_ids"] is not None else []
            scr = row["hist_scroll"] if row["hist_scroll"] is not None else []
            tm = row["hist_time"] if row["hist_time"] is not None else []
            ts = row["hist_ts"] if row["hist_ts"] is not None else []

            # Chuyển sang numpy và clean NaN
            np_ids = np.array(ids, dtype=np.int32)
            np_scr = np.nan_to_num(np.array(scr, dtype=np.float32), nan=0.0)
            np_tm = np.nan_to_num(np.array(tm, dtype=np.float32), nan=0.0)
            np_ts = np.nan_to_num(np.array(ts, dtype=np.float32), nan=0.0)

            data_map[uid] = (np_ids, np_scr, np_tm, np_ts)

        print(f"✅ History Map Loaded. Users: {len(data_map)}")
        return data_map

    def _process_row(self, row):
        user_id = row["user_id"]
        imp_ts = row["imp_ts"]
        if imp_ts is None or np.isnan(imp_ts): imp_ts = 0.0

        # --- 1. LẤY LỊCH SỬ ---
        if user_id in self.history_map:
            raw_ids, raw_scr, raw_time, raw_ts = self.history_map[user_id]
        else:
            raw_ids = np.array([], dtype=np.int32)
            raw_scr = np.array([], dtype=np.float32)
            raw_time = np.array([], dtype=np.float32)
            raw_ts = np.array([], dtype=np.float32)

        # --- 2. PADDING / TRUNCATING ---
        curr_len = len(raw_ids)
        pad_len = max(0, self.history_len - curr_len)

        def pad_arr(arr, val=0.0, dtype=np.float32):
            if curr_len >= self.history_len: return arr[-self.history_len:].astype(dtype)
            return np.pad(arr, (0, pad_len), 'constant', constant_values=val).astype(dtype)

        hist_ids = pad_arr(raw_ids, val=0, dtype=np.int64)
        hist_scr = pad_arr(raw_scr, val=0.0)
        hist_time = pad_arr(raw_time, val=0.0)

        # [DYNAMIC FEATURE 1] History Recency
        if curr_len > 0:
            # Clean raw_ts trước khi trừ
            safe_raw_ts = np.nan_to_num(raw_ts, nan=imp_ts)
            ts_diff = np.clip(imp_ts - safe_raw_ts, 0, None) / 3600.0
            ts_diff_log = np.log1p(ts_diff)
            # Clean kết quả lần nữa
            ts_diff_log = np.nan_to_num(ts_diff_log, nan=0.0)
        else:
            ts_diff_log = np.array([], dtype=np.float32)

        hist_diff = pad_arr(ts_diff_log, val=0.0)

        # --- 3. NEGATIVE SAMPLING ---
        inv_ids = row["inv_ids"] if row["inv_ids"] is not None else []
        clk_ids = row["clk_ids"] if row["clk_ids"] is not None else []

        if len(clk_ids) > 0:
            pos_id = np.random.choice(clk_ids)
        else:
            pos_id = inv_ids[0] if len(inv_ids) > 0 else 0

        neg_pool = list(set(inv_ids) - set(clk_ids))
        if not neg_pool: neg_pool = [pos_id]

        if len(neg_pool) >= self.neg_ratio:
            neg_ids = np.random.choice(neg_pool, self.neg_ratio, replace=False).tolist()
        else:
            neg_ids = (neg_pool * (self.neg_ratio // len(neg_pool) + 1))[:self.neg_ratio]

        candidate_ids = [pos_id] + neg_ids

        # --- 4. ARTICLE FEATURES LOOKUP ---
        cand_nums, cand_cats = self.art_feats.get(candidate_ids)

        # [DYNAMIC FEATURE 2] Freshness
        pub_ts = cand_nums[:, 4]
        # Clean pub_ts
        pub_ts = np.nan_to_num(pub_ts, nan=imp_ts)
        freshness = np.log1p(np.abs(imp_ts - pub_ts) / 3600.0)
        cand_nums[:, 4] = np.nan_to_num(freshness, nan=0.0)

        # --- 5. CANDIDATE SIMILARITY ---
        if curr_len > 0:
            recent_ids = raw_ids[-self.history_len:]
            hist_vecs = self.emb_manager.get_vectors_by_indices(recent_ids)
            user_vec = np.mean(hist_vecs, axis=0)
        else:
            user_vec = np.zeros(self.emb_manager.vectors.shape[1], dtype=np.float32)

        cand_vecs = self.emb_manager.get_vectors_by_indices(candidate_ids)
        scores = np.dot(cand_vecs, user_vec).reshape(-1, 1)
        scores = np.nan_to_num(scores, nan=0.0)  # Clean score

        # --- 6. IMPRESSION FEATURES ---
        time_of_day = (imp_ts % 86400) / 86400.0
        norm_age = row["norm_age"] if row["norm_age"] is not None else 0.0

        imp_feats = torch.tensor([
            np.log1p(curr_len),
            np.nan_to_num(time_of_day, nan=0.0),
            np.nan_to_num(norm_age, nan=0.0)
        ], dtype=torch.float)

        # --- RETURN DICT ---
        return {
            "hist_indices": torch.from_numpy(hist_ids),
            "hist_scroll": torch.from_numpy(hist_scr),
            "hist_time": torch.from_numpy(hist_time),
            "hist_diff": torch.from_numpy(hist_diff),

            "cand_indices": torch.tensor(candidate_ids, dtype=torch.long),
            "cand_num": torch.from_numpy(cand_nums),
            "cand_cat": torch.from_numpy(cand_cats).long(),
            "cand_sim": torch.from_numpy(scores).float(),

            "imp_feats": imp_feats,
            "device_type": torch.tensor(row["device_type"] or 0, dtype=torch.long),

            "labels": torch.tensor([1.0] + [0.0] * self.neg_ratio, dtype=torch.float)
        }

    def _stream_parquet(self):
        pf = pq.ParquetFile(self.behaviors_path)
        for batch in pf.iter_batches(batch_size=self.batch_size * 10):
            batch_dict = batch.to_pydict()
            keys = batch_dict.keys()
            length = len(next(iter(batch_dict.values())))
            for i in range(length):
                yield {k: batch_dict[k][i] for k in keys}

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        total_shards = world_size * num_workers
        current_shard = rank * num_workers + worker_id

        iterator = self._stream_parquet()
        sharded_iter = islice(iterator, current_shard, None, total_shards)
        processed_iter = (self._process_row(row) for row in sharded_iter)

        if self.shuffle_buffer > 0 and self.mode == 'train':
            buffer = []
            try:
                for _ in range(self.shuffle_buffer):
                    buffer.append(next(processed_iter))
            except StopIteration:
                pass

            while buffer:
                try:
                    new_item = next(processed_iter)
                    idx = np.random.randint(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = new_item
                except StopIteration:
                    np.random.shuffle(buffer)
                    yield from buffer
                    buffer = []
        else:
            yield from processed_iter


# ==========================================
# 4. LIGHTNING DATA MODULE (Giữ nguyên)
# ==========================================
class NAMLDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, embedding_path, batch_size=32, history_len=30, num_workers=2):
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

        common_args = {
            'article_features': self.art_feats,
            'embedding_manager': self.emb_manager,
            'history_len': self.hparams.history_len,
            'batch_size': self.hparams.batch_size,
        }

        if stage == 'fit' or stage is None:
            self.train_ds = PreprocessedIterableDataset(
                behaviors_path=self.processed_dir / "train" / "behaviors_processed.parquet",
                history_path=self.processed_dir / "train" / "history_processed.parquet",
                mode='train',
                shuffle_buffer=10000,
                **common_args
            )
            self.val_ds = PreprocessedIterableDataset(
                behaviors_path=self.processed_dir / "validation" / "behaviors_processed.parquet",
                history_path=self.processed_dir / "validation" / "history_processed.parquet",
                mode='val',
                shuffle_buffer=0,
                **common_args
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)
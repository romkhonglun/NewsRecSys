import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================
# 1. CONFIGURATION
# ==========================================
class TIME_FEATURE_NAMLConfig:
    def __init__(self):
        # --- Dimensions ---
        # Kích thước vector BERT/RoBERTa từ file body_emb.npy
        self.pretrained_dim = 1024

        # Kích thước vector chuẩn (d_model) cho toàn bộ mạng
        self.window_size = 128

        # --- Feature Engineering Specs (Khớp với dataset.py) ---
        # [log_views, log_inviews, sentiment, log_read_time, freshness]
        self.num_numerical_features = 5

        # Kích thước embedding cho mỗi feature số sau khi qua hàm Sinusoidal
        self.sinusoidal_dim = 16

        # [log_hist_len, time_of_day, norm_age]
        self.num_impression_features = 3

        # Category ID
        self.num_categories = 50  # Tùy chỉnh theo dataset thực tế
        self.category_emb_dim = 32

        # --- User Model Specs ---
        self.num_interests = 5  # Số lượng vector sở thích (K)
        self.dropout = 0.1

        # --- Ranker Specs ---
        self.rankformer_layers = 8
        self.rankformer_heads = 4
        self.rankformer_ffn_dim = 512


# ==========================================
# 2. BUILDING BLOCKS (UTILITIES)
# ==========================================
class SinusoidalEmbedding(nn.Module):
    """
    Biến đổi một giá trị số thực (scalar) thành vector (vectorization).
    Giúp mạng nơ-ron học tốt hơn các tính chất thứ tự và độ lớn.
    """

    def __init__(self, embedding_dim: int, M: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.M = M

    def forward(self, x: torch.Tensor):
        # x shape đầu vào: [Batch, Seq] hoặc [Batch]
        # Nếu x có dimension cuối là 1 [Batch, Seq, 1], cần squeeze
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        device = x.device
        half_dim = self.embedding_dim // 2

        # Công thức Positional Encoding chuẩn
        emb = np.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))

        # [Batch, Seq, 1] * [1, 1, Half_Dim] -> [Batch, Seq, Half_Dim]
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)

        # Concat sin và cos -> [Batch, Seq, Dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==========================================
# 3. ENCODERS
# ==========================================

class CandidateFeatureEncoder(nn.Module):
    """
    Mã hóa bài báo ứng viên (Candidate).
    Kết hợp: Body Vector + 5 Feature Số + 1 Feature Category.
    """

    def __init__(self, config, pretrained_vectors):
        super().__init__()
        self.config = config

        # 1. Pretrained Body Embedding (Đóng băng để tiết kiệm VRAM)
        self.body_emb = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.body_proj = nn.Sequential(
            nn.Linear(config.pretrained_dim, config.window_size),
            nn.LayerNorm(config.window_size),
            nn.GELU()
        )

        # 2. Numerical Features (5 features -> 5 Sinusoidal Layers)
        self.num_sinusoidal = nn.ModuleList([
            SinusoidalEmbedding(config.sinusoidal_dim)
            for _ in range(config.num_numerical_features)
        ])

        # 3. Categorical Feature
        self.cat_emb = nn.Embedding(config.num_categories + 1, config.category_emb_dim, padding_idx=0)

        # 4. Fusion Layer
        # Tính tổng kích thước input sau khi concat tất cả
        total_input_dim = (
                config.window_size +
                (config.num_numerical_features * config.sinusoidal_dim) +
                config.category_emb_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, config.window_size),
            nn.LayerNorm(config.window_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

    def forward(self, indices, num_feats, cat_feats):
        """
        indices:   [Batch, Seq]
        num_feats: [Batch, Seq, 5]
        cat_feats: [Batch, Seq, 1]
        """
        # A. Xử lý Body
        body_vec = self.body_emb(indices)  # [B, S, 768]
        body_vec = self.body_proj(body_vec)  # [B, S, Window_Size]

        # B. Xử lý Numerical (Sinusoidal)
        num_embs = []
        for i in range(self.config.num_numerical_features):
            col = num_feats[:, :, i]  # [B, S]
            emb = self.num_sinusoidal[i](col)  # [B, S, Sin_Dim]
            num_embs.append(emb)
        num_vec = torch.cat(num_embs, dim=-1)  # [B, S, 5 * Sin_Dim]

        # C. Xử lý Categorical
        cat_vec = self.cat_emb(cat_feats.squeeze(-1))  # [B, S, Cat_Dim]

        # D. Fusion
        concat = torch.cat([body_vec, num_vec, cat_vec], dim=-1)
        out = self.fusion(concat)  # [B, S, Window_Size]

        return out


class ImpressionEncoder(nn.Module):
    """
    Mã hóa ngữ cảnh hiển thị (Impression).
    Input: [log_hist_len, time_of_day, norm_age]
    """

    def __init__(self, config):
        super().__init__()
        self.sinusoidal = nn.ModuleList([
            SinusoidalEmbedding(config.sinusoidal_dim) for _ in range(config.num_impression_features)
        ])

        input_dim = config.num_impression_features * config.sinusoidal_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.window_size),
            nn.GELU()
        )

    def forward(self, imp_feats):
        # imp_feats: [Batch, 3]
        embs = []

        # [FIX]: Thêm len() vào đây
        for i in range(len(self.sinusoidal)):
            col = imp_feats[:, i]
            # unsqueeze(1) giả lập seq_len=1 cho Sinusoidal rồi squeeze lại
            emb = self.sinusoidal[i](col.unsqueeze(1)).squeeze(1)
            embs.append(emb)

        concat = torch.cat(embs, dim=-1)  # [Batch, 3 * Sin_Dim]
        return self.proj(concat)  # [Batch, Window_Size]


class MultiInterestUserEncoder(nn.Module):
    """
    Mã hóa lịch sử User thành K vectors sở thích.
    """

    def __init__(self, config, pretrained_vectors):
        super().__init__()
        self.dim = config.window_size
        self.num_interests = config.num_interests

        # Embedding layer (Share weight với Candidate Encoder để đồng bộ không gian vector)
        self.body_emb = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.body_proj = nn.Sequential(
            nn.Linear(config.pretrained_dim, self.dim),
            nn.GELU()
        )

        # Interaction Projection (Scroll, Time, Recency)
        # Input: Scroll(1) + Time(1) + Recency(1) = 3
        self.inter_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, self.dim)
        )
        self.layer_norm = nn.LayerNorm(self.dim)

        # Transformer Encoder để học chuỗi lịch sử
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Multi-Interest Attention Mechanism
        # Learnable Queries (Seed Vectors)
        self.interest_queries = nn.Parameter(torch.randn(self.num_interests, self.dim))
        self.interest_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, hist_indices, hist_scroll, hist_time, hist_diff, mask=None):
        # 1. Embed Content
        x = self.body_emb(hist_indices)  # [B, S, 768]
        x = self.body_proj(x)  # [B, S, Window_Size]

        # 2. Embed Interaction
        inter = torch.stack([hist_scroll, hist_time, hist_diff], dim=-1)  # [B, S, 3]
        inter = self.inter_proj(inter)

        # 3. Combine & Normalize
        x = self.layer_norm(x + inter)

        # 4. Sequence Encoding
        # mask=True là padding
        seq_rep = self.transformer(x, src_key_padding_mask=mask)

        # 5. Extract Interests
        batch_size = x.size(0)
        # Expand queries cho từng user trong batch -> [B, K, Dim]
        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Query: Interests, Key/Value: History Sequence
        interests, _ = self.interest_attn(
            query=queries,
            key=seq_rep,
            value=seq_rep,
            key_padding_mask=mask
        )
        return interests  # [Batch, K, Window_Size]


# ==========================================
# 4. MAIN MODEL (VARIANT NAML)
# ==========================================
class TIME_FEATURE_NAML(nn.Module):
    def __init__(self, config, pretrained_vectors_tensor):
        super().__init__()
        self.config = config

        # --- Init Encoders ---
        self.cand_encoder = CandidateFeatureEncoder(config, pretrained_vectors_tensor)
        self.user_encoder = MultiInterestUserEncoder(config, pretrained_vectors_tensor)
        self.imp_encoder = ImpressionEncoder(config)

        # --- Ranker Transformer ---
        # Input sequence: [Context, Interest_1...K, Cand_1...N]
        ranker_layer = nn.TransformerEncoderLayer(
            d_model=config.window_size,
            nhead=config.rankformer_heads,
            dim_feedforward=config.rankformer_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.ranker = nn.TransformerEncoder(ranker_layer, num_layers=config.rankformer_layers)

        # --- Final Prediction Head ---
        self.head = nn.Sequential(
            nn.Linear(config.window_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        """
        Input 'batch' là dict từ NAMLIterableDataset
        """
        # ==========================
        # 1. ENCODE HISTORY (USER)
        # ==========================
        hist_idx = batch["hist_indices"]
        # Mask padding: True nếu index == 0
        hist_mask = (hist_idx == 0)

        # Trả về K vectors sở thích: [Batch, K, Dim]
        user_interests = self.user_encoder(
            hist_indices=hist_idx,
            hist_scroll=batch["hist_scroll"],
            hist_time=batch["hist_time"],
            hist_diff=batch["hist_diff"],
            mask=hist_mask
        )

        # ==========================
        # 2. ENCODE CONTEXT (IMPRESSION)
        # ==========================
        # [Batch, Dim] -> [Batch, 1, Dim]
        imp_ctx = self.imp_encoder(batch["imp_feats"]).unsqueeze(1)

        # ==========================
        # 3. ENCODE CANDIDATES (ITEMS)
        # ==========================
        cand_idx = batch["cand_indices"]
        # Trả về N vectors ứng viên: [Batch, N, Dim]
        cand_vecs = self.cand_encoder(
            indices=cand_idx,
            num_feats=batch["cand_num"],
            cat_feats=batch["cand_cat"]
        )

        # ==========================
        # 4. RANKING
        # ==========================
        # Ghép thành chuỗi sequence dài: [Context, Interests, Candidates]
        # Shape: [Batch, 1 + K + N, Dim]
        seq = torch.cat([imp_ctx, user_interests, cand_vecs], dim=1)

        # Tạo Attention Mask cho Ranker
        # Context (1) + Interests (K) luôn luôn hợp lệ -> Mask = False
        batch_size = seq.size(0)
        fixed_len = 1 + self.config.num_interests
        fixed_mask = torch.zeros((batch_size, fixed_len), device=seq.device, dtype=torch.bool)

        # Candidates (N) có thể bị padding (nếu sequence < neg_ratio) -> Mask = True
        # Trong dataset.py, candidate padding là index 0
        cand_pad_mask = (cand_idx == 0)

        # Ghép mask
        full_mask = torch.cat([fixed_mask, cand_pad_mask], dim=1)

        # Transformer Interaction
        out = self.ranker(seq, src_key_padding_mask=full_mask)

        # Chỉ lấy phần output tương ứng với Candidates
        cand_out = out[:, fixed_len:, :]

        # Dự đoán điểm số
        scores = self.head(cand_out).squeeze(-1)  # [Batch, N]

        # Mask các vị trí padding thành -inf để Softmax/CrossEntropy không học
        scores = scores.masked_fill(cand_pad_mask, -65000)

        return {
            "preds": scores,
            "labels": batch.get("labels", None)
        }
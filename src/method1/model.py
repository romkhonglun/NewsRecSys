import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CONFIGURATION
# ==========================================
class VariantNAMLConfig:
    def __init__(self):
        # --- 1. Input Projector Specs ---
        # Kích thước vector từ News Encoder gốc (ví dụ BERT hoặc Pretrained NAML)
        self.embedding_dim = 1024
        # --- Multi-Interest Specs ---
        self.num_interests = 5  # Số lượng vector sở thích muốn trích xuất (K)
        # --- 2. Main Dimension (d_model) ---
        # Kích thước vector chuẩn cho toàn bộ luồng xử lý sau này
        self.window_size = 128

        # --- 3. Internal Attention ---
        self.query_vector_dim = 200
        self.dropout = 0.2
        self.num_res_blocks = 2

        # --- 4. Transformer Specs (User Encoder & Ranker) ---
        self.rankformer_layers = 8  # Số layer cho Ranker (Final Stage)
        self.rankformer_heads = 4
        self.rankformer_ffn_dim = 512

        # --- 5. Metadata Specs ---
        self.metadata_dim = 32
        self.metadata_hidden = 64
        self.num_numerical_features = 5
        self.num_categorical_features = 10


# ==========================================
# 2. BUILDING BLOCKS (TIỆN ÍCH)
# ==========================================
class SinusoidalEmbedding(nn.Module):
    """Mã hóa số thực (như Age, ReadTime) thành vector dày đặc."""

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.randn(1, output_dim // 2))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(-1)
        freqs = x.unsqueeze(-1) * self.weights
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x): return x + self.block(x)


class DeepProjector(nn.Module):
    """Nén feature từ Embedding Dim lớn (1024) xuống nhỏ (128)."""

    def __init__(self, input_dim, hidden_dim, num_res_blocks=2, dropout=0.1):
        super().__init__()
        self.compress_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.deep_stack = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)])

    def forward(self, x): return self.deep_stack(self.compress_layer(x))


class AdditiveAttention(nn.Module):
    """Attention Pooling: Gom chuỗi vector thành 1 vector."""

    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        # x: [Batch, Seq, Dim]
        proj = self.tanh(self.linear(x))
        scores = torch.matmul(proj, self.query)  # [Batch, Seq]

        if mask is not None:
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)

        weights = self.softmax(scores)
        output = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return output


# ==========================================
# 3. COMPONENT ENCODERS
# ==========================================

class InteractionAwareUserEncoder(nn.Module):
    """
    Encoder User thông minh: Kết hợp nội dung đã đọc + hành vi (Scroll, Time).
    Sử dụng Transformer để học chuỗi hành vi.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.window_size

        # 1. Project Interaction Features (Scroll + Time) -> Embedding Dim
        # Input dim = 2 (scroll, time)
        self.inter_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.dim)
        )

        # 2. Transformer Encoder cho User History
        # Dùng 2 layers để học sequence pattern ngắn hạn
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 3. Attention Pooling
        self.attn_pool = AdditiveAttention(self.dim, config.query_vector_dim)
        self.layer_norm = nn.LayerNorm(self.dim)

    def forward(self, news_vecs, scrolls, times, mask=None):
        """
        news_vecs: [Batch, Seq, Dim] - Vector nội dung bài báo
        scrolls:   [Batch, Seq]      - % cuộn trang (đã normalize)
        times:     [Batch, Seq]      - Thời gian đọc (đã log-normalize)
        """
        # Tạo feature vector từ tương tác
        inter_feats = torch.stack([scrolls, times], dim=-1)  # [B, S, 2]
        inter_emb = self.inter_proj(inter_feats)  # [B, S, Dim]

        # Fusion: Cộng Content + Interaction (Residual Connection Style)
        # Content là chính, hành vi là bổ trợ
        combined = self.layer_norm(news_vecs + inter_emb)

        # Transformer: Học ngữ cảnh chuỗi
        # src_key_padding_mask=mask (True là bị che, False là giữ)
        seq_rep = self.transformer(combined, src_key_padding_mask=mask)

        # Pooling: Gom lại thành 1 vector user
        user_vec = self.attn_pool(seq_rep, mask=mask)
        return user_vec


class MetadataEncoder(nn.Module):
    """Xử lý Metadata (Context hoặc Candidate)."""

    def __init__(self, config, has_similarity=False):
        super().__init__()
        self.config = config
        self.num_emb = SinusoidalEmbedding(config.metadata_dim)
        self.cat_proj = nn.Linear(config.num_categorical_features, config.metadata_dim)

        self.has_similarity = has_similarity
        # Tính kích thước đầu vào cho MLP tổng hợp
        input_concat_dim = config.metadata_dim * (3 if has_similarity else 2)

        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_concat_dim, config.metadata_hidden),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm1d(config.metadata_hidden)

    def forward(self, num_feats, cat_feats, sim_feats=None):
        # Numerical -> Vector
        v_num = self.num_emb(num_feats).mean(dim=-2)

        # Categorical -> Vector
        v_cat = self.cat_proj(cat_feats.float())

        if self.has_similarity and sim_feats is not None:
            if not hasattr(self, 'sim_emb'):
                self.sim_emb = SinusoidalEmbedding(self.config.metadata_dim).to(num_feats.device)
            v_sim = self.sim_emb(sim_feats).squeeze(-2)
            concat_vec = torch.cat([v_num, v_cat, v_sim], dim=-1)
        else:
            concat_vec = torch.cat([v_num, v_cat], dim=-1)

        # MLP + Batch Norm
        x = self.mlp[0](concat_vec)
        x = self.mlp[1](x)

        if x.dim() == 3:  # Xử lý trường hợp có dimension sequence
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        else:
            x = self.bn(x)

        x = self.mlp[2](x)
        return x


class MultiInterestUserEncoder(nn.Module):
    """
    Multi-Interest Encoder: Trích xuất K vector sở thích thay vì 1.
    Sử dụng cơ chế Query Attention (Seed Vectors).
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.window_size
        self.num_interests = config.num_interests

        # 1. Project Interaction Features (Giữ nguyên logic cũ)
        self.inter_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.dim)
        )

        # 2. Transformer Encoder cho History (Giữ nguyên)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(self.dim)

        # --- 3. MULTI-INTEREST EXTRACTION ---
        # Thay vì AdditiveAttention, ta dùng K learnable queries
        # Shape: [K, Dim]
        self.interest_queries = nn.Parameter(torch.randn(self.num_interests, self.dim))

        # Dùng Multihead Attention để Query vào History
        # Query = Interest Seeds, Key/Value = User History
        self.interest_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, news_vecs, scrolls, times, mask=None):
        # --- A. Contextual Interaction Encoding (Giữ nguyên) ---
        inter_feats = torch.stack([scrolls, times], dim=-1)
        inter_emb = self.inter_proj(inter_feats)
        combined = self.layer_norm(news_vecs + inter_emb)

        # seq_rep: [Batch, Seq_Len, Dim]
        seq_rep = self.transformer(combined, src_key_padding_mask=mask)

        # --- B. Multi-Interest Extraction (Mới) ---
        batch_size = seq_rep.size(0)

        # 1. Expand Queries cho từng user trong batch
        # [K, Dim] -> [Batch, K, Dim]
        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Attention: Queries tìm kiếm thông tin trong seq_rep
        # key_padding_mask=mask để Queries không chú ý vào vùng padding của lịch sử
        # Output: [Batch, K, Dim]
        user_interests, _ = self.interest_attn(
            query=queries,
            key=seq_rep,
            value=seq_rep,
            key_padding_mask=mask
        )

        return user_interests  # Trả về K vectors


class NewsEncoder(nn.Module):
    """Mã hóa nội dung bài báo (Title, Body, Cat) thành vector."""

    def __init__(self, config):
        super().__init__()
        # Giả lập Embedding layer (trong thực tế sẽ load weight pretrained)
        self.title_emb = nn.Embedding(1, config.embedding_dim)
        self.body_emb = nn.Embedding(1, config.embedding_dim)
        self.cat_emb = nn.Embedding(1, config.embedding_dim)

        self.title_proj = DeepProjector(config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout)
        self.body_proj = DeepProjector(config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout)
        self.cat_proj = DeepProjector(config.embedding_dim, config.window_size, num_res_blocks=0,
                                      dropout=config.dropout)

        self.final_attention = AdditiveAttention(config.window_size, config.query_vector_dim)

    def forward(self, indices):
        # Trong thực tế, indices này dùng để lookup pre-computed embedding
        # Ở đây demo flow
        t_vec = self.title_proj(self.title_emb(indices))
        b_vec = self.body_proj(self.body_emb(indices))
        c_vec = self.cat_proj(self.cat_emb(indices))

        batch_size, num_news, dim = t_vec.shape
        stacked = torch.stack([t_vec, b_vec, c_vec], dim=2).view(-1, 3, dim)
        news_vec = self.final_attention(stacked)
        return news_vec.view(batch_size, num_news, dim)


# ==========================================
# 4. MAIN MODEL INTEGRATION
# ==========================================
class VariantNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.news_encoder = NewsEncoder(config)

        # ==> THAY ĐỔI 1: Dùng class mới
        self.user_encoder = MultiInterestUserEncoder(config)

        # ... (giữ nguyên các phần Metadata Encoders) ...
        self.impression_meta_enc = MetadataEncoder(config, has_similarity=False)
        self.inview_meta_enc = MetadataEncoder(config, has_similarity=True)

        self.fusion_dim = config.window_size + config.metadata_hidden
        self.fusion_proj = nn.Linear(self.fusion_dim, config.window_size)

        # ... (giữ nguyên Ranker Transformer) ...
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.window_size,
            nhead=config.rankformer_heads,
            dim_feedforward=config.rankformer_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.rankformer_layers)

        self.final_mlp = nn.Sequential(
            nn.Linear(config.window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        # ... (Phần encode News giữ nguyên) ...
        hist_idx = batch['hist_indices']
        hist_scroll = batch['hist_scroll']
        hist_time = batch['hist_time']
        cand_idx = batch['cand_indices']

        hist_vecs = self.news_encoder(hist_idx)
        cand_content = self.news_encoder(cand_idx)
        hist_mask = (hist_idx == 0)

        # ==> THAY ĐỔI 2: Lấy K vectors từ User Encoder
        # Shape: [Batch, K, Dim]
        user_multi_interests = self.user_encoder(
            hist_vecs, hist_scroll, hist_time, mask=hist_mask
        )

        # ... (Encode Metadata giữ nguyên) ...
        imp_meta = self.impression_meta_enc(
            batch['imp_num'].unsqueeze(1),
            batch['imp_cat'].unsqueeze(1)
        ).squeeze(1)  # [Batch, Meta_Dim]

        cand_meta = self.inview_meta_enc(
            batch['cand_num'], batch['cand_cat'], batch['cand_sim']
        )

        # ==> THAY ĐỔI 3: Fusion logic cho Multi-Interest
        # Ta cần cộng Metadata ngữ cảnh vào TẤT CẢ K vectors sở thích
        # imp_meta: [Batch, Meta_Dim] -> [Batch, 1, Meta_Dim] -> Broadcast
        imp_meta_expanded = imp_meta.unsqueeze(1).expand(-1, self.config.num_interests, -1)

        # Concatenate: [Batch, K, Dim] + [Batch, K, Meta_Dim]
        user_fusion_input = torch.cat([user_multi_interests, imp_meta_expanded], dim=-1)

        # Chiếu về dimension chuẩn: [Batch, K, Dim]
        user_nodes = self.fusion_proj(user_fusion_input)

        # Candidate nodes (giữ nguyên): [Batch, Num_Cand, Dim]
        cand_nodes = self.fusion_proj(torch.cat([cand_content, cand_meta], dim=-1))

        # ==> THAY ĐỔI 4: Tạo chuỗi Sequence mới cho Ranker
        # Sequence bây giờ là: [Interest_1, Interest_2, ..., Interest_K, Cand_1, Cand_2, ...]
        seq = torch.cat([user_nodes, cand_nodes], dim=1)

        # Xử lý Mask cho Ranker
        # Mask cho User part là False (không che) vì Interest luôn tồn tại
        batch_size = hist_idx.size(0)

        # Tạo mask cho phần User Interests (Mặc định là False - tức là không che, vì Interest luôn valid)
        # Shape: [Batch, K]
        user_mask = torch.zeros((batch_size, self.config.num_interests),
                                device=hist_idx.device, dtype=torch.bool)

        # Kiểm tra an toàn: Chỉ nối mask nếu 'cand_mask' thực sự tồn tại
        if 'cand_mask' in batch and batch['cand_mask'] is not None:
            # batch['cand_mask'] shape: [Batch, Num_Cand]
            # full_mask shape: [Batch, K + Num_Cand]
            full_mask = torch.cat([user_mask, batch['cand_mask']], dim=1)
        else:
            # Nếu không có cand_mask (tức là mọi ứng viên đều hợp lệ), ta để None
            # Transformer của PyTorch hiểu None nghĩa là không mask gì cả
            full_mask = None

        # Transformer Interaction
        out = self.transformer(seq, src_key_padding_mask=full_mask)

        # Lấy output (Cắt bỏ K vị trí đầu là User Interests)
        cand_out = out[:, self.config.num_interests:, :]

        scores = self.final_mlp(cand_out).squeeze(-1)
        return scores
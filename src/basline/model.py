import torch
import torch.nn as nn
import torch.nn.functional as F


class NAMLConfig:
    def __init__(self):
        self.embedding_dim = 1024
        self.num_filters = 400
        self.window_size = 3
        self.query_vector_dim = 200
        self.dropout = 0.2
        self.neg_ratio = 4


class CNNFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.cnn = nn.Conv1d(
            in_channels=1,
            out_channels=config.num_filters,
            kernel_size=config.window_size,
            padding=config.window_size // 2
        )
        self.relu = nn.ReLU()
        self.attention_linear = nn.Linear(config.num_filters, config.query_vector_dim)
        self.attention_query = nn.Parameter(torch.randn(config.query_vector_dim))
        self.attention_tanh = nn.Tanh()

    def forward(self, vector_input):
        # vector_input shape: (Total_Samples, Embedding_Dim)
        # Total_Samples = Batch_Size * Sequence_Length

        # 1. Unsqueeze để tạo channel dimension cho Conv1d
        # (N, EmbDim) -> (N, 1, EmbDim)
        x = vector_input.unsqueeze(1)
        x = self.dropout(x)

        # 2. CNN
        # (N, 1, EmbDim) -> (N, NumFilters, EmbDim)
        x = self.relu(self.cnn(x))

        # 3. Permute để tính attention
        # (N, NumFilters, EmbDim) -> (N, EmbDim, NumFilters)
        x = x.permute(0, 2, 1)

        # 4. Attention Pooling
        proj = self.attention_tanh(self.attention_linear(x))
        scores = torch.matmul(proj, self.attention_query)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)

        # Output: (N, NumFilters)
        output = torch.sum(x * weights, dim=1)
        return output


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        proj = self.tanh(self.linear(x))
        scores = torch.matmul(proj, self.query)
        weights = self.softmax(scores)
        output = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return output


class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.title_emb = nn.Embedding(1, config.embedding_dim)
        self.body_emb = nn.Embedding(1, config.embedding_dim)
        self.cat_emb = nn.Embedding(1, config.embedding_dim)

        self.title_cnn = CNNFeatureExtractor(config)
        self.body_cnn = CNNFeatureExtractor(config)

        self.cat_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.num_filters),
            nn.ReLU()
        )

        self.final_attention = AdditiveAttention(config.num_filters, config.query_vector_dim)

    def forward(self, news_indices):
        """
        Input: news_indices (Batch, Seq_Len) HOẶC (Batch, )
        """
        # Lưu lại shape ban đầu để reshape sau này
        original_shape = news_indices.shape

        # 1. Flatten inputs: (Batch, Seq_Len) -> (Batch * Seq_Len)
        # Nếu input đã là 1D (Batch,) thì view(-1) vẫn giữ nguyên đúng logic
        flat_indices = news_indices.view(-1)

        # 2. Embedding Lookup
        t_vec = self.title_emb(flat_indices)  # (Total, EmbDim)
        b_vec = self.body_emb(flat_indices)
        c_vec = self.cat_emb(flat_indices)

        # 3. Feature Extraction (CNN xử lý từng bài báo một)
        t_rep = self.title_cnn(t_vec)  # (Total, NumFilters)
        b_rep = self.body_cnn(b_vec)
        c_rep = self.cat_proj(c_vec)

        # 4. Multi-view Attention
        stacked_views = torch.stack([t_rep, b_rep, c_rep], dim=1)
        news_vectors = self.final_attention(stacked_views)  # (Total, NumFilters)

        # 5. Reshape về shape ban đầu
        # (Total, NumFilters) -> (Batch, Seq_Len, NumFilters)
        if len(original_shape) > 1:
            return news_vectors.view(original_shape[0], original_shape[1], -1)
        else:
            return news_vectors  # (Batch, NumFilters)


class UserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AdditiveAttention(config.num_filters, config.query_vector_dim)

    def forward(self, clicked_vecs):
        return self.attention(clicked_vecs)


class OriginalNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)

    def forward(self, hist_indices, cand_indices):
        # hist_indices: (Batch, Hist_Len)
        # cand_indices: (Batch, Cand_Len)

        # 1. Encode News (Model tự xử lý flatten bên trong)
        hist_vecs = self.news_encoder(hist_indices)  # -> (Batch, Hist_Len, Dim)
        cand_vecs = self.news_encoder(cand_indices)  # -> (Batch, Cand_Len, Dim)

        # 2. Encode User
        user_vec = self.user_encoder(hist_vecs)  # -> (Batch, Dim)

        # 3. Score
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(2)).squeeze(2)
        return scores
"""
饮料销售预测 - Transformer时间序列预测模型
用于预测未来销售数据，支持生产优化决策
"""

import torch
import torch.nn as nn
import math


# 位置编码层
class PositionalEncoding(nn.Module):
    """
    位置编码：为序列中的每个位置添加位置信息
    使用正弦和余弦函数生成位置编码
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 多头自注意力层
class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出变换
        output = self.W_o(context)

        return output, attn_weights


# 前馈神经网络
class FeedForward(nn.Module):
    """
    前馈神经网络层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器单层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


# Transformer解码器层
class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器单层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 交叉注意力
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x


# 饮料销售预测Transformer模型
class SalesForecasterTransformer(nn.Module):
    """
    饮料销售预测Transformer模型

    用于预测未来销售数据，输入历史销售序列，输出预测的未来销售量

    参数:
        input_dim: 输入特征维度（如：5种饮料 + 时间特征等）
        d_model: 模型隐藏维度
        num_heads: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        d_ff: 前馈网络隐藏维度
        input_seq_len: 输入序列长度（历史天数）
        output_seq_len: 输出序列长度（预测天数）
        dropout: Dropout比率
    """
    def __init__(self, input_dim=5, d_model=128, num_heads=8,
                 num_encoder_layers=4, num_decoder_layers=4, d_ff=512,
                 input_seq_len=30, output_seq_len=7, dropout=0.1):
        super(SalesForecasterTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # 输入嵌入层
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(input_seq_len, output_seq_len) + 100, dropout=dropout)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, input_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size):
        """生成用于解码器的因果掩码（防止看到未来信息）"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def encode(self, src):
        """编码器前向传播"""
        # 嵌入 + 位置编码
        x = self.encoder_embedding(src)
        x = self.pos_encoder(x)

        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def decode(self, tgt, encoder_output, tgt_mask=None):
        """解码器前向传播"""
        # 嵌入 + 位置编码
        x = self.decoder_embedding(tgt)
        x = self.pos_encoder(x)

        # 通过解码器层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask)

        return x

    def forward(self, src, tgt):
        """
        前向传播

        Args:
            src: 源序列（历史数据） (batch_size, input_seq_len, input_dim)
            tgt: 目标序列（用于训练时的teacher forcing） (batch_size, output_seq_len, input_dim)

        Returns:
            output: 预测的销售数据 (batch_size, output_seq_len, input_dim)
        """
        # 生成解码器掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # 编码
        encoder_output = self.encode(src)

        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask)

        # 输出投影
        output = self.output_layer(decoder_output)

        return output

    def predict(self, src, predict_len=None):
        """
        自回归预测（推理时使用）

        Args:
            src: 历史数据 (batch_size, input_seq_len, input_dim)
            predict_len: 预测长度，默认为output_seq_len

        Returns:
            predictions: 预测结果 (batch_size, predict_len, input_dim)
        """
        if predict_len is None:
            predict_len = self.output_seq_len

        self.eval()
        device = src.device
        batch_size = src.size(0)

        with torch.no_grad():
            # 编码历史数据
            encoder_output = self.encode(src)

            # 初始化解码器输入（使用历史数据的最后一个时间步）
            decoder_input = src[:, -1:, :]  # (batch_size, 1, input_dim)

            predictions = []

            for _ in range(predict_len):
                # 解码
                tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                decoder_output = self.decode(decoder_input, encoder_output, tgt_mask)

                # 获取最后一个时间步的预测
                next_pred = self.output_layer(decoder_output[:, -1:, :])
                predictions.append(next_pred)

                # 更新解码器输入
                decoder_input = torch.cat([decoder_input, next_pred], dim=1)

            # 合并所有预测
            predictions = torch.cat(predictions, dim=1)

        return predictions


# 简化版编码器-仅模型（用于快速训练）
class SalesForecasterEncoderOnly(nn.Module):
    """
    仅使用编码器的销售预测模型（更快速训练）

    直接将历史序列编码后通过全连接层输出预测
    """
    def __init__(self, input_dim=5, d_model=128, num_heads=8,
                 num_layers=4, d_ff=512, input_seq_len=30,
                 output_seq_len=7, dropout=0.1):
        super(SalesForecasterEncoderOnly, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # 输入嵌入
        self.embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_seq_len + 100, dropout=dropout)

        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层：将整个序列映射到预测结果
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * input_seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_seq_len * input_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        """
        前向传播

        Args:
            src: 历史销售数据 (batch_size, input_seq_len, input_dim)

        Returns:
            output: 预测结果 (batch_size, output_seq_len, input_dim)
        """
        batch_size = src.size(0)

        # 嵌入 + 位置编码
        x = self.embedding(src)
        x = self.pos_encoder(x)

        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x)

        # 展平并输出
        x = x.reshape(batch_size, -1)
        output = self.output_layer(x)
        output = output.reshape(batch_size, self.output_seq_len, self.input_dim)

        return output

    def predict(self, src, predict_len=None):
        """推理预测"""
        self.eval()
        with torch.no_grad():
            return self.forward(src)


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型参数（适配饮料销售预测场景）
    # 5种饮料：碳酸饮料、果汁饮料、茶饮料、功能饮料、矿泉水
    input_dim = 5
    input_seq_len = 30   # 使用过去30天数据
    output_seq_len = 7   # 预测未来7天

    # 创建完整Transformer模型
    model_full = SalesForecasterTransformer(
        input_dim=input_dim,
        d_model=128,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=512,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        dropout=0.1
    ).to(device)

    # 创建简化版模型
    model_simple = SalesForecasterEncoderOnly(
        input_dim=input_dim,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        dropout=0.1
    ).to(device)

    print("\n=== 完整Transformer模型 ===")
    print(f"模型参数量: {sum(p.numel() for p in model_full.parameters()):,}")

    print("\n=== 简化版编码器模型 ===")
    print(f"模型参数量: {sum(p.numel() for p in model_simple.parameters()):,}")

    # 测试前向传播
    batch_size = 4
    src = torch.randn(batch_size, input_seq_len, input_dim).to(device)
    tgt = torch.randn(batch_size, output_seq_len, input_dim).to(device)

    # 完整模型测试
    output_full = model_full(src, tgt)
    print(f"\n完整模型输出形状: {output_full.shape}")  # 应为 (4, 7, 5)

    # 简化模型测试
    output_simple = model_simple(src)
    print(f"简化模型输出形状: {output_simple.shape}")  # 应为 (4, 7, 5)

    # 测试预测功能
    predictions = model_full.predict(src)
    print(f"预测输出形状: {predictions.shape}")  # 应为 (4, 7, 5)

    print("\n模型测试通过！")

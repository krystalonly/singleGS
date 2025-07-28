import torch.nn as nn
import torch
import torch.nn.functional as F

from scene.transformer.layer_norm import LayerNorm
from scene.transformer.multi_head_attention import MultiHeadAttention
from scene.transformer.position_wise_feed_forward import PositionwiseFeedForward


class Spatial_Audio_Attention_Layer(nn.Module):
    def __init__(self, args):
        super(Spatial_Audio_Attention_Layer, self).__init__()
        self.args = args

        self.enc_dec_attention = MultiHeadAttention(d_model=self.args.d_model, n_head=self.args.n_head)

        self.norm1 = LayerNorm(d_model=self.args.d_model)
        self.dropout1 = nn.Dropout(p=self.args.drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=self.args.d_model, hidden=self.args.ffn_hidden,
                                           drop_prob=self.args.drop_prob)

        self.norm2 = LayerNorm(d_model=self.args.d_model)
        self.dropout2 = nn.Dropout(p=self.args.drop_prob)

    def forward(self, x, enc_source):
        _x = x

        # enc_source = self.sl(x, enc_source)
        x, att = self.enc_dec_attention(q=x, k=enc_source, v=enc_source, mask=None)

        # 4. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        # x = self.norm2(x)

        return x, att


class Spatial_Audio_Attention_Module(nn.Module):
    def __init__(self, args):
        super(Spatial_Audio_Attention_Module, self).__init__()
        self.args = args
        self.layers = nn.ModuleList([Spatial_Audio_Attention_Layer(args) for _ in range(args.n_layer)])

    def forward(self, x, enc_source):
        attention = []
        for layer in self.layers:
            x, att = layer(x, enc_source)
            attention.append(att.mean(dim=1).unsqueeze(dim=1))
        attention = torch.cat(attention, dim=1)  # B, layer, N, 3
        return x, attention


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.query_linear = nn.Linear(feature_dim, feature_dim)
        self.key_linear = nn.Linear(feature_dim, feature_dim)
        self.value_linear = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = (feature_dim // num_heads) ** 0.5

        self.offset_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, enc_source):
        # 生成查询、键、和值
        Q = self.query_linear(x)  # 音频特征生成查询
        K = self.key_linear(enc_source)  # 面部和视角特征生成键
        V = self.value_linear(enc_source)  # 面部和视角特征生成值

        # 计算注意力分数
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # 转换成多头格式
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权和值
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(out.size(0), Q.size(2), -1)

        x_ = self.offset_linear(out)
        x = x + x_
        return x, attention_weights


class CrossModalAttentionModule(nn.Module):
    """
    交叉模态注意力模块，用于初始化模型
    """

    def __init__(self, args):
        super(CrossModalAttentionModule, self).__init__()
        self.cross_modal_attention = CrossModalAttention(feature_dim=args.d_model, num_heads=args.n_head)

    def forward(self, x, enc_source):
        """
        调用 CrossModalAttention 实现模态融合
        """
        x, attention_weights = self.cross_modal_attention(x, enc_source)
        return x, attention_weights


class MultiModalFusion(nn.Module):
    def __init__(self, feature_dim):
        super(MultiModalFusion, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, enc_source):
        enc_source = self.norm(F.relu(self.fc(enc_source)))
        return enc_source



import torch 
import torch.nn as nn
from sublayers import MultiHeadAttention, PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop):
        """ 
        x - Multi-head attention - FFN 
        residual & layer norm已经在模块内实现 
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, p_drop)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ffn, p_drop)
    
    def forward(self, 
                padding_mask: torch.Tensor,
                x: torch.Tensor):
        x = self.multi_head_attention(padding_mask, x)
        x = self.feed_forward(x)
        return x 

class DecoderLayer(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(num_heads, d_model, p_drop)
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, p_drop)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ffn, p_drop)
    
    def forward(self,
                padding_mask: torch.Tensor,
                mask: torch.Tensor,
                x_decoder: torch.Tensor,
                x_encoder: torch.Tensor):
        """ 
        mask: 对于decoder的掩码, 以确保不会获取'未来'信息
        padding_mask: 对于encoder的掩码, 不考虑padding位置
        x_decoder: 来自上一层Decoder Layer / 上一次预测的词嵌入
        x_encoder: 来自最后一层Encoder Layer
        """
        x_decoder = self.masked_multi_head_attention(mask, x_decoder)
        # encoder输出作为K, V; decoder输出作为Q
        output = self.multi_head_attention(padding_mask, x_encoder, x_decoder)
        output = self.feed_forward(output)
        return output 
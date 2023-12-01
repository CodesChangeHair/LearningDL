from turtle import forward
import torch 
import torch.nn as nn 
from attention import ScaledDotProductAttention

class PositionWiseFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        """ 
        linear layer -- relu -- linear layer -- residual -- layer norm
        d_model: MHA输出向量长度
        d_ffn: 中间隐藏层维度
        """       
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, x: torch.Tensor):
        residual = x 
       
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        
        # residual & layer norm
        output = output + residual 
        return torch.layer_norm(output, normalized_shape=output.size()[1:])

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 p_drop: float=0.1):
        """ 
        num_heads: 注意力头的数量
        d_model: 输入向量长度 
        """
        super(MultiHeadAttention, self).__init__()
        # 多个注意力并行计算, 以达到多通道输出, 为计算效率考虑每个注意力头的维度是 d_model / num_heads
        self.multi_head_attention = nn.ModuleList(
            ScaledDotProductAttention(d_model,
                                      d_model // num_heads,
                                      d_model // num_heads)
            for _ in range(num_heads)
        )
        # 从value空间线性投影至输出空间 
        self.wo = nn.Parameter(torch.zeros(d_model, d_model))
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self,
                mask: torch.Tensor,
                x_key_value: torch.Tensor,
                x_query: torch.Tensor=None):
        
        attention = torch.concatenate([
            atten(mask, x_key_value, x_query)
            for atten in self.multi_head_attention
        ], dim=2)
        
        output = attention @ self.wo
        
        # residual(Q) & layer norm
        output += x_key_value if x_query is None else x_query
        return torch.layer_norm(output, normalized_shape=output.size()[1:])


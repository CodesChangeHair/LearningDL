from turtle import forward
import torch
import torch.nn as nn
import math


NEG_INF = float("-inf")

class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_model: int, 
                 d_query_key: int,
                 d_value: int):
        """ 
        通过scaled的点积操作计算q, k向量之间的相似度作为注意力分数 
        d_model: 输入特征向量的长度. 网络大量使用residual connection, 多数向量特征长度相同
        d_query_key: 单个query, key向量的长度
        d_value: 单个value向量的长度  
        """
        super(ScaledDotProductAttention, self).__init__()
        # Q, K, V: 线性投影矩阵, 将输入I投影至各自空间
        self.wk = nn.Parameter(torch.zeros(d_model, d_query_key))
        self.wq = nn.Parameter(torch.zeros(d_model, d_query_key))
        self.wv = nn.Parameter(torch.zeros(d_model, d_value))
        self.div = math.sqrt(d_query_key)  # scaled factor
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self,
                mask: torch.Tensor,
                x_key_value: torch.Tensor,
                x_query: torch.Tensor=None):
        """ 
        mask: bool Tensor, 为true的地方表示允许分配注意力
        """
        # 对于Encoder, Q, K, V均来自自身输入
        # 对于Decoder, K, V可能来自Encoder(cross attention)
        if x_query is None:
            x_query = x_key_value
        # Q, K, V = W @ I
        k = x_key_value @ self.wk 
        v = x_key_value @ self.wv 
        q = x_query @ self.wq 
        # batch matrix multiply: A = Q @ K.transpose()
        attention = torch.einsum('nik, njk -> nij', q, k) / self.div
        attention = torch.where(mask, attention, NEG_INF)  # padding mask or masked attention
        # A' = softmax(A)
        attention = self.softmax(attention)
        # batch matrix multiply: O = A' @ V
        output = torch.einsum('nik, nkj -> nij', attention, v)
        return output 
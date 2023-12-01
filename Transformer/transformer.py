from turtle import forward
import torch 
import torch.nn as nn 
from layers import EncoderLayer, DecoderLayer
from embed import TokenEmbedding, PositionEmbedding

class Encoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        """ 
        Encoder: N x Encoder Layer 堆叠N次Encoder Layer
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(num_heads, d_model, d_ffn, p_drop)
            for _ in range(num_layers)
        )
    
    def forward(self,
                padding_mask: torch.Tensor,
                src: torch.Tensor):
        for layer in self.layers:
            src = layer(padding_mask, src)
        return src 

class Decoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        """ 
        Decoder: N x Decoder Layer 堆叠N次Decoder Layer
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(num_heads, d_model, d_ffn, p_drop)
            for _ in range(num_layers)
        )
    
    def forward(self, 
                padding_mask: torch.Tensor,
                mask: torch.Tensor,
                tgt: torch.Tensor,
                encoder_out: torch.Tensor):
        for layer in self.layers:
            tgt = layer(padding_mask, mask, tgt, encoder_out)
        return tgt 

class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layer: int,
                 num_decoder_layer: int,
                 num_heads: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1,
                 max_num_words: int=None):
        """ 
        src_vocab_size: 翻译任务中原文单词库大小
        tgt_vocab_size: 翻译任务中译文单词库大小
        d_model: 词嵌入维度, 由于使用residual connection, 模型
        中大多数向量长度均为d_model
        max_num_words: 单个句子(sample)包含token的最大数量, 用于初始化Position Embedding
        """
        super(Transformer, self).__init__()
        self.src_tok_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_embed = TokenEmbedding(tgt_vocab_size, d_model)
        if max_num_words is None:
            self.pos_embed = PositionEmbedding(d_model, p_drop=p_drop)
        else:
            self.pos_embed = PositionEmbedding(d_model, max_num_words, p_drop)
        
        assert d_model % num_heads == 0 
        
        self.encoder = Encoder(num_encoder_layer, num_heads, d_model, d_ffn, p_drop)
        self.decoder = Decoder(num_decoder_layer, num_heads, d_model, d_ffn, p_drop)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, 
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor, 
                src: torch.Tensor,
                tgt: torch.Tensor):
        """ 
        src: 经过tokenize的原文
        tgt: 经过tokenize的译文
        return: 译文下一个单词的概率分布
        """
        src_embed = self.pos_embed(self.src_tok_embed(src))
        tgt_embed = self.pos_embed(self.tgt_tok_embed(tgt))
        encoder_padding_mask = src_padding_mask.expand(-1, src.size()[1], -1)
        encoder_out = self.encoder(encoder_padding_mask, src_embed)
        decoder_mask = torch.tril(tgt_padding_mask.expand(-1, tgt.size()[1], -1))
        decoder_padding_mask = src_padding_mask.expand(-1, tgt.size()[1], -1)
        decoder_out = self.decoder(decoder_padding_mask, decoder_mask, tgt_embed, encoder_out)
        return self.linear(decoder_out)
            
        
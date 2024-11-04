import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int, vocab_size:int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)   # embedding is multiplied by d_model according to the original paper.

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout:float):
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    # create a tensor (seq_len x d_model) filled with zeros as a placeholder for positional encoding
    pe = torch.zeros(seq_len, d_model)

    # numerator term of the formula for positional encoding
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    # denominator term of the formula
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # log and exp is used for numerical stability.

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # adding a batch size dimension to positional encoding, specifying that this positional encoding is used for single ip sequence
    pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

    # saving the positional encoding info as a constant buffer that is not updated during training.
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x.shape[1] gives the length of ip sequqnce.
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)

class LayerNorm(nn.Module):
  def __init__(self, eps):
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))  # default initial value is chosen as 1
    self.bias = nn.Parameter(torch.zeros(1))  # default initial value is chosen as 0

  def forward(self, x):
    mean = x.mean(-1, keepdims = True)
    std = x.std(-1, keepdims = True)
    return ((x - mean) / (std + self.eps)) * self.alpha + self.bias


class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float):
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = nn.Linear(self.d_model, self.d_ff)
    x = nn.ReLU(x)
    x = self.dropout(x)
    x = nn.Linear(self.d_ff, self.d_model)
    x = self.dropout(x)
    return x

class MultiheadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float):  # h is the number of heads
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model is not divisible by h"

    self.d_k = d_model // h
    self.w_k = nn.Linear(d_model, d_model)   # nn.Linear has default weight initialization as learnable parameter
    self.w_q = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)   # d_v * h is d_model
    self.dropout = nn.Dropout(dropout)

  # Declaring self_attention as a static method avoids the need to create an instance of the class to use it
  @staticmethod
  def selfAttention(query, key, value, mask, dropout):
    d_k = query.shape[-1]
    # @ represents matrix multiplication
    attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_len, seq_len)
    if mask is not None:
      attention_score = attention_score.masked_fill(mask == 0, -1e9)   # if mask == 0, then replace by -1e9
    attention_score = torch.softmax(attention_score, dim=-1)  # the result of multiplication across a sequence is present in the last dimension
    if dropout is not None:
      attention_score = dropout(attention_score)
    # att_score @ value is the output of self_attention
    return (attention_score @ value), attention_score

  def forward(self, q, k, v, mask):   # mask is used to mask attention of some words with other words
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)  # (batch, seq, d_model)

    # .view() method is used to change the shape of the data
    # .transpose(1,2) swaps two positions 1 and 2..we do this to change into desirable format of (seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)   # (batch, h, seq_len, d_k)

    x, attention_scores = MultiheadAttention.selfAttention(query, key, value, mask, self.dropout)

    # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

    return self.w_o(x)

class ResidualConnection(nn.Module):
  def __init__(self, dropout: float):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNorm()

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
  def __init__(self, self_attentionBlock : MultiheadAttention, feed_forwardBlock : FeedForward, dropout: float):
    super().__init__()
    self.self_attentionBlock = self_attentionBlock
    self.feed_forwardBlock = feed_forwardBlock

    self.residualConnection1 = ResidualConnection(dropout)
    self.residualConnection2 = ResidualConnection(dropout)

  def forward(self, x, src_mask):
    x = self.residualConnection1(x, lambda x : self.self_attentionBlock(x, x, x, src_mask))   # q, k and v is the input 'x' in case of the encoder
    x = self.residualConnection2(x, self.feed_forwardBlock)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, layers : nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = LayerNorm()

  def forward(self, x, src_mask):
    for layer in self.layers:
      x = layer(x, src_mask)
    return self.norm(x)

class DecoderBlock(nn.Module):
  def __init__(self, self_attention : MultiheadAttention, cross_attention : MultiheadAttention, feed_forward: FeedForward, dropout: float) -> None:
    super().__init__()
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    self.feed_forward = feed_forward

    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])   # 3 residual connections

    def forward(self, x, encoder_output, src_mask, tgt_mask):
      x = self.residual_connections[0](x, lambda x : self.self_attention(x, x, x, tgt_mask))
      x = self.residual_connections[1](x, lambda x : self.cross_attention(x, encoder_output, encoder_output, src_mask))  # cross attention layer
      x = self.residual_connections[2](x, self.feed_forward)
      return x

class DecoderLayer(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = LayerNorm

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask)
    return self.norm(x)

class LinearLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    return torch.log_softmax(self.proj(x), dim=-1)  # (Batch_size, seq_len, vocab_size)

class Transformers (nn.Module):
  def __init__(self, encoder: EncoderLayer, decoder: DecoderLayer, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_position: PositionalEncoding, tgt_position: PositionalEncoding, Linear: LinearLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_position = src_position
    self.tgt_position = tgt_position
    self.linear = Linear

    def encode(self, x, src_mask):
      x = self.src_embed(x)
      x = self.src_position(x)
      return self.encoder(x, src_mask)

    def decode(self, x, encoder_output, src_mask, tgt_mask):
      x  = self.tgt_embed(x)
      x = self.tgt_position(x)
      return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def linear_proj (self, x):
      return self.linear(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff : int = 2048):

  # building the embedding layers
  src_embed = InputEmbeddings(d_model, src_vocab_size)
  tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

  # building the positional embedding layers
  src_position = PositionalEncoding(d_model, src_seq_len, dropout)
  tgt_position = PositionalEncoding(d_model, tgt_seq_len, dropout)

  # building the encoder layer
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention = MultiheadAttention(d_model, h, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
    encoder_blocks.append(encoder_block)

  encoder = EncoderLayer(nn.ModuleList(encoder_blocks))

  # building the decoder layer
  decoder_blocks = []
  for _ in range(N):
    decoder_self_attention = MultiheadAttention(d_model, h, dropout)
    decoder_cross_attention = MultiheadAttention(d_model, h, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
    decoder_blocks.append(decoder_block)

  decoder = DecoderLayer(nn.ModuleList(decoder_blocks))

  # building the linear layer
  linear = LinearLayer(d_model, tgt_vocab_size)

  # building the transformer
  transformer = Transformers(encoder, decoder, src_embed, tgt_embed, src_position, tgt_position, linear)

  # initialize parameters
  for p in transformer.parameters():   # it returne all the learable parameters
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer
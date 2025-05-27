import torch
from torch import nn
import math

# Input Embedddings
# Creating a class for Input Embeddings
class InputEmbeddings(nn.Module):
  def __init__(self, vocab_size: int , d_model: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model)

#In the embedding layers, we multiply those weights by √dmodel
  def forward(self, x : torch.tensor) -> torch.tensor: # forward method
    return self.embedding(x) * math.sqrt(self.d_model)


# Positonal Encodings

class PositionalEncodings(nn.Module):
  def __init__(self, d_model : int, seq_len : int, dropout : float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    # CREATE A MATRIX OF (seq_length,d_model)
    pe = torch.zeros(seq_len,d_model)
    # creating the require input for postional encoding formula
    # create a vector of shape (seq_len,1)
    position = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) # we use this function because we simply the formula of the postional encoding
    # sin and cos function
    pe[:,0::2] = math.sin(position * div_term)
    pe[:,1::2] = math.cos(position * div_term)
    # since we have (sen_len,d_model) which is one batch, we need this for more sentence, we add one more dim to the matrix
    pe = pe.unsqueeze(0) # (1,seq_len,d_model)

    # we need to save this PE, so that it wont change in furture. We use buffer for this.
    self.register_buffer('pe',pe)

    def forward(self,x):
      x = x + (self.pe[:,:x.shape(1),:] )# x.shape(1) -> seq_len , adding the PE to x
      return self.dropout(x)  # to generalize the optimized value and randomly sets some values in x to zero during training.

  # End of positional encoding

  # Layer Normalization or Add and Norms
  # common terms alpha(gamma) and bias(beta), here we use beta and gamma

  class LayerNormalization(nn.Module):

    def __init__(self, eps : float = 10**-6):
      super().__init__()
      self.eps = eps
      self.alpha = nn.parameter(torch.ones(1)) #multiplicative
      self.beta = nn.parameter(torch.zeros(1)) #additive

    def forward(self,x):
      mean = x.mean(dim = -1,keepdim = True) # we use dim as keyword, because it is a default parameter and we can only use math lib for constant/scalar not on tensors.
      std = x.std(dim = -1, keepdim = True) # here dim = -1 defines the d_model
      return self.alpha * (x-mean) / (std + self.eps) + self.beta

  #end of LN

  # Feed forward layer

  class FeedForward(nn.Module): # w1 and b1 then RELU(rectified near unit) then w2 and b2
    def __init__(self,d_model : int, d_ff : int , dropout : float):
      super().__init__()
      self.linear_1 = nn.Linear(d_model,d_ff) # w1 and b1 (bias is default) #nn.Linear(d_model,d_ff,bias = True)
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
      #(batch, seq_len, d_model) -> #(batch, seq_len, d_ff) ->  #(batch, seq_len, d_model)
      return self.linear_2(self.dropout(torch.relu(self.linear_1)))

  #End of the feedforward layer

  #Multi-head Attention

  class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model : int, h : int, dropout : float) -> None:
      super().__init__()
      self.d_model = d_model
      self.h = h
      self.dropout = nn.Dropout(dropout)
      assert d_model % h == 0, "d_model must be divisible by h"  #

      #d_k = d_model//h
      self.d_k = d_model // h
      #denoting the wq, wk and wv
      self.wq = nn.Linear(d_model, d_model)
      self.wk = nn.Linear(d_model, d_model)
      self.wv = nn.Linear(d_model, d_model)
      self.wo = nn.Linear(d_model, d_model) # denoting the output matrix, d_model // h = dk  and d_model = dk * h
      self.dropout = nn.Dropout(dropout)

      @staticmethod
      def attention(query,key,value,mask,dropout: nn.Dropout):
        #(batch,h,seq_len,d_k) * (batch,h,d_k,seq_len)
        d_k = query.shape[-1]
        #(batch,h, seq_len , d_k) -> (Batch, h, seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
          attention_scores.masked_fill_(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
          attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


      def forward(self,q,k,v,mask):
        query = self.wq(q) # multiplying -> (batch,seq_len,d_model) * (batch,d_model,d_model) = (batch,seq_len,d_model)
        key = self.wk(k) # multiplying -> (batch,seq_len,d_model) * (batch,d_model,d_model) = (batch,seq_len,d_model)
        value = self.wv(v) # multiplying -> (batch,seq_len,d_model) * (batch,d_model,d_model) = (batch,seq_len,d_model)

        #(batch,seq_len,d_model) -> (batch,seq_len,h,d_k) -> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value, mask, self.dropout)
        # (Batch , h, Seq_Len, d_k) -> (Batch, Seq_Len, h, d_k) -> (Batch , seq_len, h, d_k)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        # output matirx
        #(batch,seq_len,d_model) -> (batch,seq_len,d_model)
        return self.w_o(x)

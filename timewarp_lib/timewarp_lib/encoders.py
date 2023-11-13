import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import numpy as np

class FeedForwardLayer(nn.Module):
    def __init__(self,
            in_dim,
            mid_dim,
            out_dim,
            nonlinearity,
            **kwargs):
        super(FeedForwardLayer, self).__init__()
        self.hidden_linear0 = torch.nn.Linear(in_dim, mid_dim)
        self.hidden_linear1 = torch.nn.Linear(mid_dim, out_dim)
        self.nonlinearity = nonlinearity

    def forward(self,layer):
      layer = self.hidden_linear0(layer)
      layer = self.nonlinearity(layer)
      layer = self.hidden_linear1(layer)
      return layer

# A pytorch module to take in an input of shape
# (batch_size, traj_len, traj_channels)
# and of shape 
# (batch_size, traj_len, 1)
# and output 
# 1) an embedding of the whole trajectory
#      in the shape
#     (batch_size, latent_dim)
# and also
# 2) an associated logvar for noise associated with the embedding also
#      in the shape
#     (batch_size, latent_dim)
class SelfAttentionTransformerEncoder(nn.Module):
    def __init__(self,
            latent_dim,
            traj_len,
            traj_channels,
            enc_attention_dims_per_head,
            enc_attention_num_heads = 4,
            enc_feed_forward_dim = 200,
            emb_nonlinearity="ELU",
            dtype=torch.float,
            enc_append_time_dim = False,
            **kwargs):
        super(SelfAttentionTransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.traj_channels = int(traj_channels)
        self.attention_dims_per_head = enc_attention_dims_per_head
        self.append_time_dim = enc_append_time_dim

        if emb_nonlinearity == "ReLU":
          self.nonlinearity = nn.ReLU()
        elif emb_nonlinearity == "Softplus":  
          self.nonlinearity = nn.Softplus()
        elif emb_nonlinearity == "ELU": 
          self.nonlinearity = nn.ELU()
        elif emb_nonlinearity == "Tanh": 
          self.nonlinearity = nn.Tanh()
        else:
          raise Exception(f"Unknown nonlinearity of '{emb_nonlinearity}'")
        
        # input dim is the number of channels (plus the concatenated time dimension if we're appending time)
        input_dim = self.traj_channels
        if self.append_time_dim:
            time_dim = 1
            input_dim += time_dim

        num_heads = enc_attention_num_heads
        self.attention_embed_dim = self.attention_dims_per_head * num_heads

        self.input_ff = FeedForwardLayer(input_dim, enc_feed_forward_dim, self.attention_embed_dim,self.nonlinearity)
        
        self.attention = torch.nn.MultiheadAttention(embed_dim = self.attention_embed_dim,
            num_heads=num_heads, dropout=0.0, bias=True,
            add_bias_kv=False, add_zero_attn=False,
            kdim=self.attention_embed_dim, 
            vdim=self.attention_embed_dim, 
            batch_first=False,
            dtype=dtype)

        self.mid_ff = FeedForwardLayer(self.attention_embed_dim, enc_feed_forward_dim, self.attention_embed_dim,self.nonlinearity)
        
        self.attention2 = torch.nn.MultiheadAttention(embed_dim = self.attention_embed_dim,
            num_heads=num_heads, dropout=0.0, bias=True,
            add_bias_kv=False, add_zero_attn=False,
            kdim=self.attention_embed_dim, 
            vdim=self.attention_embed_dim, 
            batch_first=False,
            dtype=dtype)

        self.mu_ff = FeedForwardLayer(self.attention_embed_dim, enc_feed_forward_dim, self.latent_dim,self.nonlinearity)
        self.logvar_ff = FeedForwardLayer(self.attention_embed_dim, enc_feed_forward_dim, self.latent_dim,self.nonlinearity)

        self.layer_norm = torch.nn.LayerNorm(self.attention_embed_dim)
        
        self.probe_query_lin = torch.nn.Linear(in_features=1,
            out_features=self.attention_embed_dim,dtype=dtype)

    # x is expected to be a Tensor of the form
    # (batch_size, traj_len, traj_channels)
    # scaled ts is expected to be a Tensor
    # (batch_size, traj_len, 1)
    # and the output is a pair of Tensors of sizes
    # batchsize x self.latent_dim
    # and
    # batchsize x self.latent_dim
    def encode(self,x,scaled_ts=None):
      xbatch_size, xtraj_len, xtraj_channels = x.shape
      if self.append_time_dim:
          tbatch_size, ttraj_len, ttraj_channels = scaled_ts.shape
          assert xtraj_channels == self.traj_channels, f"input data had {xtraj_channels} channels but should have had {self.traj_channels}"
          assert ttraj_channels == 1, f"input data had {ttraj_channels} channels for time, but should have just been 1"
          assert xtraj_len == ttraj_len, f"time and space did not align in length"
          assert xbatch_size == tbatch_size, f"time and space did not align in batch size"

      device = x.device

      if self.append_time_dim:
        train_dat_input = torch.cat((x,scaled_ts), dim=2)
      else:
        train_dat_input = x
      # Transformer wants input to be 
      # training data to be times, batchid, channels
      # so we transpose first 2 indices of x
      train_dat_input = train_dat_input.transpose(0,1)
      layer = self.input_ff(train_dat_input)

      # apply the attention layer
      layer1, _ = self.attention(layer, layer, layer, 
          need_weights=False)
      layer2 = layer1 + layer
      layer = self.layer_norm(layer2)

      layer1 = self.mid_ff(layer)
      layer2 = layer1 + layer
      layer = self.layer_norm(layer2)


      probe_vector = torch.ones((1,xbatch_size,1),dtype=x.dtype).to(device)
      probe_query = self.probe_query_lin(probe_vector)
      layer1, _ = self.attention2(probe_query, layer, layer, 
          need_weights=False)
      layer2 = layer1 + probe_query 
      layer = self.layer_norm(layer2)

      # we compute only a single output from the attention layer
      # by querying it with a linear transformation of (1,1,1,1,1,1...,1)
      # that is, the output from the attention layer is 
      # of shape (1, num_batches, attention_embed_dim)
      # because we pass in 1 query
      # (the first dim is the number of different queries)
      # and because each query returns an output of dimension attention_embed_dim
      # (the last dim is the dimension of the values in our attention network, which we
      # set to attention_embed_dim)


      # make it shape (1, batches, latent_dim)
      mu_too_large = self.mu_ff(layer) 
      # make it shape (batches, latent_dim)
      mu = mu_too_large.squeeze(0) 

      # see comments above for how this modifies the shape
      logvar_too_large = self.logvar_ff(layer)
      logvar = logvar_too_large.squeeze(0) 

      return(mu,logvar)

# A pytorch module to take in an input of shape
# (batch_size, traj_len, traj_channels)
# and of shape 
# (batch_size, traj_len, 1)
# and output 
# 1) an embedding of the whole trajectory
#      in the shape
#     (batch_size, latent_dim)
# and also
# 2) an associated logvar for noise associated with the embedding also
#      in the shape
#     (batch_size, latent_dim)
class TransformerEncoder(nn.Module):
    def __init__(self,
            latent_dim,
            traj_len,
            traj_channels,
            enc_attention_dims_per_head,
            enc_attention_num_heads=4,
            dtype=torch.float,
            enc_append_time_dim = False,
            **kwargs):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.traj_channels = int(traj_channels)
        self.attention_dims_per_head = enc_attention_dims_per_head
        self.append_time_dim = enc_append_time_dim

        num_heads = enc_attention_num_heads
        self.attention_embed_dim = self.attention_dims_per_head * num_heads


        self.attention = torch.nn.MultiheadAttention(embed_dim = self.attention_embed_dim,
            num_heads=num_heads, dropout=0.0, bias=True,
            add_bias_kv=False, add_zero_attn=False,
            kdim=self.attention_embed_dim, 
            vdim=self.attention_embed_dim, 
            batch_first=False,
            dtype=dtype)


        # input dim is the number of channels (plus the concatenated time dimension if we're appending time)
        input_dim = self.traj_channels
        if self.append_time_dim:
            time_dim = 1
            input_dim += time_dim

        self.key_lin = torch.nn.Linear(in_features=input_dim,
            out_features=self.attention_embed_dim,dtype=dtype)
        self.value_lin = torch.nn.Linear(in_features=input_dim,
            out_features=self.attention_embed_dim,dtype=dtype)
        self.mu_lin = torch.nn.Linear(in_features=self.attention_embed_dim,
            out_features=latent_dim,dtype=dtype)
        self.logvar_lin = torch.nn.Linear(in_features=self.attention_embed_dim,
            out_features=latent_dim,dtype=dtype)
        # I haven't really thought about what I want here,
        # but for now we probe the transformer with the standard basis of R^n
        # and then linearly combine the results
        self.query_combiner = torch.nn.Linear(in_features=self.attention_embed_dim,
            out_features=1,dtype=dtype)

        # INITIALIZATION CHANGE: don't initialize logvar to average to 0
        # Instead, initialize to average to -3
        self.logvar_lin.bias.data.sub_(3.)

    # x is expected to be a Tensor of the form
    # (batch_size, traj_len, traj_channels)
    # scaled ts is expected to be a Tensor
    # (batch_size, traj_len, 1)
    # and the output is a pair of Tensors of sizes
    # batchsize x self.latent_dim
    # and
    # batchsize x self.latent_dim
    def encode(self,x,scaled_ts=None):
      xbatch_size, xtraj_len, xtraj_channels = x.shape
      if self.append_time_dim:
          tbatch_size, ttraj_len, ttraj_channels = scaled_ts.shape
          assert xtraj_channels == self.traj_channels, f"input data had {xtraj_channels} channels but should have had {self.traj_channels}"
          assert ttraj_channels == 1, f"input data had {ttraj_channels} channels for time, but should have just been 1"
          assert xtraj_len == ttraj_len, f"time and space did not align in length"
          assert xbatch_size == tbatch_size, f"time and space did not align in batch size"

      device = x.device
      query = torch.eye(self.attention_embed_dim,dtype=x.dtype).unsqueeze(1).repeat(1,xbatch_size,1).to(device)

      if self.append_time_dim:
        train_dat_input = torch.cat((x,scaled_ts), dim=2)
      else:
        train_dat_input = x
      # Transformer wants input to be 
      # training data to be times, batchid, channels
      # so we transpose first 2 indices of x
      train_dat_input = train_dat_input.transpose(0,1)
      key = self.key_lin(train_dat_input)
      value = self.value_lin(train_dat_input)
      #key = torch.nn.functional.relu(self.key_lin(train_dat_input))
      #value = torch.nn.functional.relu(self.value_lin(train_dat_input))

      # apply the attention layer
      attn_output, attn_output_weights = self.attention(query, key, value, 
          need_weights=False)
      # we compute multiple outputs from the attention layer
      # that is, the output from the attention layer is 
      # of shape (1, num_batches, attention_embed_dim)
      # because we pass in 1 query
      # (the first dim is the number of different queries)
      # and because each query returns an output of dimension attention_embed_dim
      # (the last dim is the dimension of the values in our attention network, which we
      # set to attention_embed_dim)

      # make it shape (embed_dim, batches, latent_dim)
      mu_too_large = self.mu_lin(attn_output) 
      # make it shape (latent_dim, batches, embed_dim)
      mu_too_large = mu_too_large.transpose(0,2) 
      # make it shape (latent_dim, batches, 1)
      mu_transposed = self.query_combiner(mu_too_large)
      # make it shape (batches,latent_dim)
      mu = mu_transposed.squeeze(2).t()

      # see comments above for how this modifies the shape
      logvar_too_large = self.logvar_lin(attn_output)
      logvar_too_large = logvar_too_large.transpose(0,2)
      logvar = self.query_combiner(logvar_too_large).squeeze(2).t()

      return(mu,logvar)

# A pytorch module to take in an input of shape
# (batch_size, traj_len, traj_channels)
# and output 
# 1) an embedding of the whole trajectory
#      in the shape
#     (batch_size, latent_dim)
# and also
# 2) an associated logvar for noise associated with the embedding also
#      in the shape
#     (batch_size, latent_dim)
class OneDConvEncoder(nn.Module):
    def __init__(self,
            latent_dim,
            traj_len,
            traj_channels,
            emb_nonlinearity = "ReLU",
            emb_dropout_probability = 0.0,
            emb_conv_layers_channels = [],
            emb_conv_layers_strides = [],
            emb_conv_layers_kernel_sizes = [],
            emb_fc_layers_num_features = [],
            emb_activate_last_layer = False,
            emb_conv1d_padding = 0,
            dtype=torch.float,
            **kwargs):
        super(OneDConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.traj_len = int(traj_len)
        self.traj_channels = int(traj_channels)
        self.emb_conv_layers_channels = emb_conv_layers_channels
        self.emb_conv_layers_strides = emb_conv_layers_strides
        self.emb_conv_layers_kernel_sizes = emb_conv_layers_kernel_sizes
        self.emb_fc_layers_num_features = emb_fc_layers_num_features
        self.emb_dropout_probability = np.array(emb_dropout_probability).item()
        self.emb_conv1d_padding = np.array(emb_conv1d_padding).item()
        self.emb_activate_last_layer = np.array(emb_activate_last_layer).item()

        prev_channels = traj_channels
        traj_len = self.traj_len
        layer_channels = self.traj_channels
        # construct the parameters for all the embedding convolutions
        self.emb_convs = []
        for i, layer_channels in enumerate(self.emb_conv_layers_channels):
            self.emb_convs.append(
                    nn.Conv1d(prev_channels,
                              layer_channels,
                              self.emb_conv_layers_kernel_sizes[i],
                              self.emb_conv_layers_strides[i],
                              padding = self.emb_conv1d_padding,
                              dtype=dtype
                              ))
            if self.emb_conv1d_padding == "same":
              traj_len = traj_len
            else: 
              traj_len = int(math.floor(
                (traj_len - (self.emb_conv_layers_kernel_sizes[i]- 1) - 1)/
                  self.emb_conv_layers_strides[i] 
                + 1))
            prev_channels = layer_channels
        # construct the parameters for all the embedding fully-connected layers 
        self.emb_fcs = []
        # layer channels is the last-used layer channels
        prev_features = int(traj_len * layer_channels)
        layer_features = prev_features
        for layer_features in self.emb_fc_layers_num_features:
            self.emb_fcs.append(nn.Linear(prev_features, layer_features,dtype=dtype))
            prev_features = layer_features
        # separately save the mu and logvar layers
        self.fcmu = nn.Linear(layer_features, self.latent_dim,dtype=dtype)
        self.fclogvar = nn.Linear(layer_features, self.latent_dim,dtype=dtype)
        # INITIALIZATION CHANGE: don't initialize logvar to average to 0
        # Instead, initialize to average to -3
        self.fclogvar.bias.data.sub_(3.)

        self.dropout = nn.Dropout(self.emb_dropout_probability)

        emb_nonlinearity = np.array(emb_nonlinearity).item()
        if emb_nonlinearity == "ReLU":
          self.nonlinearity = nn.ReLU()
        elif emb_nonlinearity == "Softplus":  
          self.nonlinearity = nn.Softplus()
        elif emb_nonlinearity == "ELU": 
          self.nonlinearity = nn.ELU()
        elif emb_nonlinearity == "Tanh": 
          self.nonlinearity = nn.Tanh()
        else:
          raise Exception(f"Unknown nonlinearity of '{emb_nonlinearity}'")


        self.emb_convs = nn.ModuleList(self.emb_convs)
        self.emb_fcs = nn.ModuleList(self.emb_fcs)

    # x is expected to be a Tensor of the form
    # batchsize x self.traj_len x self.traj_channels
    # and the output is a pair of Tensors of sizes
    # batchsize x self.latent_dim
    # and
    # batchsize x self.latent_dim
    def encode(self,x, scaled_ts=None):
        xbatch_size, xtraj_len, xtraj_channels = x.shape
        assert xtraj_channels == self.traj_channels, f"input data had {xtraj_channels} channels but should have had {self.traj_channels}"
        assert xtraj_len == self.traj_len, f"input data had {xtraj_len} timesteps but should have had {self.traj_len}"
        # you can totally just ignore the scaled_ts
        # we don't actually use it for anything

        # NOTE: Unlike Transformer, we want the axes to be
        # batchsize x self.traj_channels x self.traj_len
        # when doing 1D convolution
        # (see https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
        x = torch.transpose(x,1,2)
        # NOTE THE SWITCHED ORDERING HERE
        xbatch_size, xtraj_channels, xtraj_len = x.shape
        assert xtraj_channels == self.traj_channels, f"input data had {xtraj_channels} channels but should have had {self.traj_channels}"
        assert xtraj_len == self.traj_len, f"input data had {xtraj_len} timesteps but should have had {self.traj_len}"

        layer = x
        for conv in self.emb_convs:
            layer = self.dropout(layer)
            layer = self.nonlinearity(conv(layer))
        # flatten all but the 0th dimension
        layer = torch.flatten(layer, 1)
        for fc in self.emb_fcs:
            layer = self.nonlinearity(fc(layer))

        mu = self.fcmu(layer)
        logvar = self.fclogvar(layer)

        if self.emb_activate_last_layer:
          mu = self.nonlinearity(mu)
          logvar = self.nonlinearity(logvar)
        return(mu,logvar)

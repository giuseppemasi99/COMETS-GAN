
import math
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from model.csdi.conv1d_with_init import Conv1dWithInit

class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1dWithInit(channels, 2 * channels)
        self.output_projection = Conv1dWithInit(channels, 2 * channels)

        self.is_linear = is_linear
        if is_linear:            
            self.time_layer = LinearAttentionTransformer(
                dim=channels, depth=1, heads=nheads, max_seq_len=256, n_local_attn_heads=0, local_attn_window_size=0
            )
            self.time_layer = LinearAttentionTransformer(
                dim=channels, depth=1, heads=nheads, max_seq_len=256, n_local_attn_heads=0, local_attn_window_size=0
            )
        else:
            self.time_layer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=channels, nhead=nheads, dim_feedforward=64, activation="gelu"), 
                num_layers=1)
            self.feature_layer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=channels, nhead=nheads, dim_feedforward=64, activation="gelu"), 
                num_layers=1
            )

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        # _, cond_dim, _, _ = cond_info.shape
        # cond_info = cond_info.reshape(B, cond_dim, K * L)
        # cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        # y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip



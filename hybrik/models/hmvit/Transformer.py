import numpy as np
from typing import Union, Tuple, List, Optional
from functools import partial


import pytorch_lightning as pl

import torch
import torch.nn as nn
import pickle
import trimesh
from smplx import SMPLX
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
import torch.nn.functional as F
from hybrik.models.hmvit.NormalPredictionNetwork import SMPLSideViewFeatureExtractor
#from NormalPredictionNetwork import SMPLSideViewFeatureExtractor



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: (int, int) of the grid height and width
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim]
    """
    grid_h, grid_w = grid_size
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.stack([grid_w.repeat(grid_h.size), np.tile(grid_h, grid_w.size)])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out =np.outer(pos,omega)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))



class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x).contiguous(), **kwargs).contiguous()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x).contiguous()
        x = self.act(x).contiguous()
        x = self.drop(x).contiguous()
        x = self.conv2(x).contiguous()
        x = self.drop(x).contiguous()
        return x.contiguous()

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x).contiguous()



def split_chunks(x, chunk_size):
    """
    进行分块注意力的先前准备
    """
    return [x[:, :, i:i+chunk_size, :].contiguous() for i in range(0, x.size(2), chunk_size)]

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,chunk_size:int =64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.chunk_size = chunk_size

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        torch.cuda.empty_cache()  # 清理未使用的显存

        # 分块处理
        q_chunks = split_chunks(q, self.chunk_size)
        k_chunks = split_chunks(k, self.chunk_size)
        v_chunks = split_chunks(v, self.chunk_size)

        attn_chunks = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            attn_chunk = (q_chunk @ k_chunk.transpose(-2, -1)).contiguous() * self.scale
            attn_chunk = self.attend(attn_chunk).contiguous()
            attn_chunks.append((attn_chunk @ v_chunk).contiguous())

        out = torch.cat(attn_chunks, dim=2).contiguous()
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

        return self.to_out(out).contiguous()

class FrontAttention(nn.Module):
    """
    Front view attention module, used to extract the features of the front view in the image.
    This module implements a layer with a multi-head attention mechanism
    and reduces video memory usage by block processing.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,chunk_size=64):
        super(FrontAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2).contiguous()  # 变换为 (B, N, C)
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()

        torch.cuda.empty_cache()  # 清理未使用的显存
        # 分块处理
        q_chunks = split_chunks(q, self.chunk_size)
        k_chunks = split_chunks(k, self.chunk_size)
        v_chunks = split_chunks(v, self.chunk_size)

        attn_chunks = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            attn_chunk = (q_chunk @ k_chunk.transpose(-2, -1)).contiguous() * self.scale
            attn_chunk = attn_chunk.softmax(dim=-1).contiguous()
            attn_chunk = self.attn_drop(attn_chunk).contiguous()
            attn_chunks.append((attn_chunk @ v_chunk).contiguous())

        attn = torch.cat(attn_chunks, dim=2).contiguous()

        x = attn.transpose(1, 2).contiguous().view(B, -1, C).contiguous()
        x = self.proj(x).contiguous()
        x = self.proj_drop(x).contiguous()
        x = x.transpose(1, 2).contiguous().view(B, C, H, W).contiguous()  # 还原为 (B, C, H, W)

        return x.contiguous()

class CVA_Attention(nn.Module):
    """
    交叉注意力模块，主要是为了提取其他几个视图相对于前视图特征提取不足的地方，
    为了最后的融合模块进行特征的融合
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,chunk_size =64):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.chunk_size =chunk_size

        self.Qnorm = nn.LayerNorm(dim)
        self.Knorm = nn.LayerNorm(dim)
        self.Vnorm = nn.LayerNorm(dim)
        self.QLinear = nn.Linear(dim, dim)
        self.KLinear = nn.Linear(dim, dim)
        self.VLinear = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, CVA_input):
        # Reshape and permute
        B, C, H, W = x.shape
        N = H * W
        x = x.view(B, C, N).transpose(1, 2).contiguous()  # 变换为 (B, N, C)
        CVA_input = CVA_input.reshape(CVA_input.shape[0], -1, CVA_input.shape[1]).contiguous()  # (B, N, C)

        q = self.QLinear(self.Qnorm(CVA_input).contiguous()).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = self.KLinear(self.Knorm(CVA_input).contiguous()).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = self.VLinear(self.Vnorm(x).contiguous()).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        torch.cuda.empty_cache()  # 清理未使用的显存

        # 分块处理
        q_chunks = split_chunks(q, self.chunk_size)
        k_chunks = split_chunks(k, self.chunk_size)
        v_chunks = split_chunks(v, self.chunk_size)

        attn_chunks = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            attn_chunk = (q_chunk @ k_chunk.transpose(-2, -1)).contiguous() * self.scale
            attn_chunk = attn_chunk.softmax(dim=-1).contiguous()
            attn_chunk = self.attn_drop(attn_chunk).contiguous()
            attn_chunks.append((attn_chunk @ v_chunk).contiguous())

        attn = torch.cat(attn_chunks, dim=2).contiguous()
        x = attn.transpose(1, 2).contiguous().view(B, N, C).contiguous()

        x = self.proj(x).contiguous()
        x = self.proj_drop(x).contiguous()

        return x.transpose(1, 2).contiguous().view(B, C, H, W).contiguous()

class Front_Out_Block(nn.Module):
    """
    The output of the front view is used to fuse the features of different views into the front view.
    The output of the front view, the obtained MSA, is used as the key q and k in the cross attention
    to match the features in other views.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer([64, 64])
        self.attn = FrontAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop) #前向注意力层
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([64, 64])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        MSA = self.drop_path(self.attn(self.norm1(x))).contiguous()
        x = (x + MSA).contiguous()
        x = (x + self.drop_path(self.mlp(self.norm2(x))).contiguous()).contiguous()
        return x.contiguous(), MSA.contiguous()

class Multi_In_Out_Block(nn.Module):
    """
    This part is mainly used to calculate the offset of
    several other views relative to the front view
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.cva_attn = CVA_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.norm1 = norm_layer([64, 64])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, CVA_input):
        MSA = self.drop_path(self.cva_attn(x, CVA_input).contiguous()).contiguous()
        x = (x + MSA).contiguous()

        x = (x + self.drop_path(self.mlp(self.norm1(x).contiguous())).contiguous()).contiguous()
        return x.contiguous()
    

class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,chunk_size: int =64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.chunk_size =chunk_size

        self.attend = nn.Softmax(dim=-1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.norm = nn.LayerNorm(dim)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.multi_head_attention = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head))

    def forward(self, x: torch.FloatTensor, q_x: torch.FloatTensor) -> torch.FloatTensor:
        q_in = self.multi_head_attention(q_x).contiguous() + q_x.contiguous()
        q_in = self.norm(q_in).contiguous()

        q = rearrange(self.to_q(q_in).contiguous(), 'b n (h d) -> b h n d', h=self.heads).contiguous()
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads).contiguous(), kv)

        attn = torch.matmul(q, k.transpose(-1, -2)).contiguous() * self.scale
        attn = self.attend(attn).contiguous()

        out = torch.matmul(attn, v).contiguous()
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

        return self.to_out(out).contiguous(), q_in.contiguous()

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x).contiguous() + x.contiguous()
            x = ff(x).contiguous() + x.contiguous()

        return self.norm(x).contiguous()

class CrossTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([CrossAttention(dim, heads=heads, dim_head=dim_head),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor, q_x: torch.FloatTensor) -> torch.FloatTensor:
        encoder_output = x
        for attn, ff in self.layers:
            x, q_in = attn(encoder_output, q_x)
            x = x.contiguous() + q_in.contiguous()
            x = ff(x).contiguous() + x.contiguous()
            q_x = x.contiguous()

        return self.norm(q_x).contiguous()

class ViTEncoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches  # 192

        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))


        self.patch_dim = channels * patch_height * patch_width  ##768

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0),
                                             requires_grad=False)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img).contiguous()
        assert x.size(1) == self.num_patches, f"Expected {self.num_patches} patches, but got {x.size(1)}"
        x = x.contiguous() + self.en_pos_embedding[:, :x.size(1), :].contiguous()
        x = self.transformer(x).contiguous()

        return x.contiguous()

class ViTDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 32, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
            else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
            else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0),
                                             requires_grad=False)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4)
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token.contiguous() + self.de_pos_embedding.contiguous()
        x = self.transformer(x).contiguous()
        x = self.to_pixel(x).contiguous()

        return x.contiguous()

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight

class CrossAttDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, input_channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = num_patches
        self.patch_dim = input_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0),
                                             requires_grad=False)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim, input_channels, kernel_size=4, stride=4)
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor, query_img: torch.FloatTensor) -> torch.FloatTensor:
        B, N, D = token.size()
        assert N == self.num_patches, f"Expected {self.num_patches} patches, but got {N} patches."
        query = self.to_patch_embedding(query_img).contiguous()
        assert query.size(1) == self.num_patches, f"Expected query to have {self.num_patches} patches, but got {query.size(1)} patches."
        query = query.contiguous() + self.de_pos_embedding[:, :N, :].contiguous()
        x = token.contiguous() + self.de_pos_embedding[:, :N, :].contiguous()
        x = self.transformer(x.contiguous(), query.contiguous()).contiguous()
        x = self.to_pixel(x).contiguous()
        return x.contiguous()

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight

class ViTVQ(pl.LightningModule):
    def __init__(self, image_size=256, patch_size=16, channels=3) -> None:
        super().__init__()


        # image_size = (x.shape[2], x.shape[3])
        self.smpl_feature_extractor =SMPLSideViewFeatureExtractor(input_channels=32)
            
        self.encoder = ViTEncoder(image_size=image_size, patch_size=patch_size, dim=256, depth=4, heads=4, mlp_dim=1024,
                                  channels=channels)
        self.F_decoder = ViTDecoder(image_size=image_size, patch_size=patch_size, dim=256, depth=2, heads=4,
                                    mlp_dim=1024)
        self.B_decoder = CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256, depth=2, heads=4,
                                         mlp_dim=1024)
        self.R_decoder = CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256, depth=2, heads=4,
                                         mlp_dim=1024)
        self.L_decoder = CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256, depth=2, heads=4,
                                         mlp_dim=1024)
        self.Front_decoder = Front_Out_Block(dim=32, num_heads=4,qkv_bias=True)
        self.dc =conv1x1(3, 32)
        self.Multi_decoder =Multi_In_Out_Block(dim=32, num_heads=4,qkv_bias=True)


    def forward(self, x: torch.FloatTensor,input_features) -> torch.FloatTensor:
        enc_out = self.encode(x).contiguous()           #[2,256,256] 准确来说应该是[b,256,256] 
        smpl_normal =self.smpl_feature_extractor(input_features=input_features)
        dec = self.decode(enc_out.contiguous(),smpl_normal)
        dec = self.fuse(dec).contiguous()

        return dec.contiguous()

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        h = self.encoder(x).contiguous()
        return h.contiguous()  # , emb_loss

    def decode(self, enc_out: torch.FloatTensor, smpl_normal) -> torch.FloatTensor:
        back_query = smpl_normal['T_normal_B'].contiguous()
        right_query = smpl_normal['T_normal_R'].contiguous()
        left_query = smpl_normal['T_normal_L'].contiguous()
        # quant = self.post_quant(quant)
        dec_F = self.F_decoder(enc_out.contiguous()).contiguous()
        dec_B = self.B_decoder(enc_out.contiguous(), back_query.contiguous()).contiguous()
        dec_R = self.R_decoder(enc_out.contiguous(), right_query.contiguous()).contiguous()
        dec_L = self.L_decoder(enc_out.contiguous(), left_query.contiguous()).contiguous()
        dec_B =self.dc(dec_B)
        dec_R =self.dc(dec_R)
        dec_L =self.dc(dec_L)

        return (dec_F.contiguous(), dec_B.contiguous(), dec_R.contiguous(), dec_L.contiguous())

    def fuse(self, dec):

        dec = list(dec)
        Front_x, Front_MSA = self.Front_decoder(dec[0].contiguous())

        #解码不对的视图

        Back_x = self.Multi_decoder(dec[1].contiguous(), Front_MSA.contiguous()).contiguous()
        Right_x = self.Multi_decoder(dec[2].contiguous(), Front_MSA.contiguous()).contiguous()
        Left_x = self.Multi_decoder(dec[3].contiguous(), Front_MSA.contiguous()).contiguous()

        # fusion
        x = (Front_x.contiguous() + Back_x.contiguous() + Right_x.contiguous() + Left_x.contiguous()).contiguous()

        return x.contiguous()



if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    if x.is_cuda:
        print("张量在 GPU 上")
    else:
        print("张量在 CPU 上")
    print("张量 x 在设备:", x.device)
    model =ViTVQ()

    input_features = torch.randn(1, 32, 64, 64)
    dec = model(x, input_features)
    print(dec.shape)
    print(dec.is_contiguous())



# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model

        # div_term: [d_model // 2] 只需生成一次，重复使用
        self.div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model))

    def forward(self, x):
        B, C, H, W = x.shape

        # y_position: [H, 1]
        y_position = torch.arange(0, H, dtype=torch.float, device=x.device).unsqueeze(1).repeat(1, W)

        # x_position: [1, W]
        x_position = torch.arange(0, W, dtype=torch.float, device=x.device).unsqueeze(0).repeat(H, 1)

        # 动态生成正弦和余弦位置编码
        pe = torch.zeros(C, H, W, device=x.device)

        # 确保 div_term 移动到与 x 相同的设备上
        div_term = self.div_term.to(x.device).unsqueeze(-1).unsqueeze(-1)

        # 正弦编码 (y方向)
        pe[0:C // 2:2, :, :] = torch.sin(y_position.unsqueeze(0) * div_term)

        # 余弦编码 (x方向)
        pe[1:C // 2:2, :, :] = torch.cos(x_position.unsqueeze(0) * div_term)

        # 添加位置编码到输入
        x = x + pe.unsqueeze(0)  # [B, C, H, W]

        return x

class PoseAttention(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=4, chunk_size=64):  # 减小 chunk_size
        super(PoseAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.chunk_size = chunk_size

        # 输入投影层，将输入的通道数映射到 d_model 维度
        self.input_proj = nn.Conv2d(input_dim, d_model, kernel_size=1) if input_dim != d_model else nn.Identity()

        # 位置编码
        self.pos_encoder = PositionalEncoding2D(d_model)

        # 线性层用于生成 query, key, value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 最终输出的线性变换
        self.fc = nn.Linear(d_model, d_model)

    def split_chunks(self, x, chunk_size):
        """
        将输入张量分块
        """
        B, T, C = x.shape
        return x.view(B, T // chunk_size, chunk_size, C).transpose(1, 2).contiguous().view(-1, chunk_size, C)

    def forward(self, pose_features):
        # 输入是 [B, C, H, W]
        B, C, H, W = pose_features.shape

        # 1. 将输入通道数映射到 d_model 维度
        x = self.input_proj(pose_features)  # [B, d_model, H, W]

        # 2. 添加位置编码
        x = self.pos_encoder(x)  # [B, d_model, H, W]

        # 3. 分块处理空间维度以减少计算量
        chunk_size = self.chunk_size
        H_chunks = H // chunk_size
        W_chunks = W // chunk_size

        x = x.view(B, self.d_model, H_chunks, chunk_size, W_chunks, chunk_size)  # [B, d_model, H_chunks, chunk_size, W_chunks, chunk_size]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, self.d_model, chunk_size * chunk_size)  # [B * H_chunks * W_chunks, d_model, chunk_size * chunk_size]
        x = x.permute(0, 2, 1)  # [B * H_chunks * W_chunks, chunk_size * chunk_size, d_model]

        # 4. 线性映射并分块处理
        q_chunks = self.split_chunks(self.w_q(x), chunk_size)  # [num_chunks, chunk_size, d_model]
        k_chunks = self.split_chunks(self.w_k(x), chunk_size)  # [num_chunks, chunk_size, d_model]
        v_chunks = self.split_chunks(self.w_v(x), chunk_size)  # [num_chunks, chunk_size, d_model]

        attn_chunks = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            attn_chunk = (q_chunk @ k_chunk.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)))  # [num_chunks, chunk_size, chunk_size]
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            attn_chunk = torch.matmul(attn_chunk, v_chunk)  # [num_chunks, chunk_size, d_k]
            attn_chunks.append(attn_chunk)

        # 将所有分块的注意力输出拼接
        attn_output = torch.cat(attn_chunks, dim=0)  # [B * H_chunks * W_chunks, chunk_size, d_model]

        # 5. 恢复空间维度
        attn_output = attn_output.view(B, H_chunks, W_chunks, chunk_size, chunk_size, self.d_model)  # [B, H_chunks, W_chunks, chunk_size, chunk_size, d_model]
        attn_output = attn_output.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, d_model, H, W]
        output = attn_output.view(B, self.d_model, H, W)  # [B, d_model, H, W]
        output = self.fc(output.permute(0, 2, 3, 1))  # [B, H, W, d_model]
        output = output.permute(0, 3, 1, 2)  # [B, d_model, H, W]

        return output

    

class PoseEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.1):
        super(PoseEncoderLayer, self).__init__()
        self.self_attn = PoseAttention(input_dim=d_model, d_model=d_model, n_heads=n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # 输入是 [B, C, H, W]
        B, C, H, W = src.shape
        src_flat = src.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, d_model]

        # Self attention layer
        src2 = self.self_attn(src)  # [B, d_model, H, W]
        src2_flat = src2.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, d_model]
        src_flat = src_flat + self.dropout1(src2_flat)
        src_flat = self.norm1(src_flat)

        # Feed-forward layer
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_flat))))
        src_flat = src_flat + self.dropout2(src2)
        src_flat = self.norm2(src_flat)

        # 恢复空间维度
        src = src_flat.permute(0, 2, 1).contiguous(memory_format=torch.contiguous_format).view(B, C, H, W)  # [B, d_model, H, W]
        return src

class TransformerRefine(nn.Module):
    def __init__(self, input_dim=17, d_model=256, n_heads=4, num_layers=4, d_ff=1024, dropout=0.1):
        super(TransformerRefine, self).__init__()
        self.input_proj = nn.Conv2d(input_dim, d_model, kernel_size=1)
        self.encoder_layers = nn.ModuleList([PoseEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Conv2d(d_model, input_dim, kernel_size=1)

    def forward(self, pose_features):
        # 输入投影到 d_model 维度
        x = self.input_proj(pose_features)  # [B, d_model, H, W]

        # 通过多个编码层
        for layer in self.encoder_layers:
            x = layer(x)  # [B, d_model, H, W]

        # 输出投影回原始通道数
        refined_pose = self.output_proj(x)  # [B, 17, H, W]

        return refined_pose
    

def get_pose_refine(input_dim=17, d_model=256, n_heads=4, num_layers=4, d_ff=1024, dropout=0.1):
    """
    Returns a TransformerRefine model instance for pose feature refinement.

    Parameters:
    input_dim (int): Number of input feature channels, default is 17.
    d_model (int): Dimension of features in encoder, default is 256.
    n_heads (int): Number of heads for multi-head attention mechanism, default is 4.
    num_layers (int): Number of layers for encoder, default is 4.
    d_ff (int): Hidden layer size for feedforward neural network, default is 1024.
    dropout (float): Dropout ratio, default is 0.1.

    Returns:
    TransformerRefine model instance.
    """
    model =TransformerRefine(input_dim=input_dim, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)

    return model
    

if __name__ == '__main__':
    import time
    start_time = time.time()
    pose_features = torch.randn(1, 17, 64, 64)  # 假设输入数据: batch size 1，17 个通道（关节点数），64x64 尺寸
    # model = PoseAttention(input_dim=17).cuda()
    # output = model(pose_features)
    # print(output.shape)  # 输出应为 [1, 256, 64, 64]，其中 256 是 d_model 的值

    # print(output.is_cuda)  # 返回 True 表示在 GPU 上，False 表示在 CPU 上

    # # 测试 PoseEncoderLayer
    # encoder_layer = PoseEncoderLayer().cuda()
    # encoder_output = encoder_layer(output)
    # print(encoder_output.shape)  # 输出应为 [1, 256, 64, 64]
    # print(encoder_output.is_cuda)  # 返回 True 表示在 GPU 上，False 表示在 CPU 上

    # # 测试 TransformerRefine
    # transformer_refine = TransformerRefine().cuda()
    # refined_output = transformer_refine(pose_features)
    # print(refined_output.shape)  # 输出应为 [1, 17, 64, 64]

    model =get_pose_refine(input_dim=17)
    output = model(pose_features)

    end_time = time.time()
    print("time:",end_time-start_time)

    print(output.shape)
    
    print(output.is_cuda)

    #判断是否连续
    # print(output.is_contiguous())  # 返回 True 表示连续存储，False 表示不连续存储
    # print(encoder_output.is_contiguous())  # 返回 True 表示连续存储，False 表示不连续存储
    print(output.is_contiguous())  # 返回 True 表示连续存储，False 表示不连续存储
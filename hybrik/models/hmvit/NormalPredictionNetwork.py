import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalPredictionNetwork(nn.Module):
    def __init__(self, input_channels, output_channels=3):
        """
        法线预测网络，从输入特征直接生成法线特征图。
        :param input_channels: 输入特征的通道数
        :param output_channels: 输出法线特征的通道数（默认 3）
        """
        super(NormalPredictionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # 输出大小为 [B, 3, H, W]
        # 归一化法线方向
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-8
        x = x / norm  # 每个像素的法线方向归一化
        return x.contiguous()  # 保证内存连续性


class SideViewFeatureExtractor(nn.Module):
    def __init__(self, input_channels, feature_channels=32, output_channels=3):
        """
        基于视角掩码提取侧视图特征。
        :param input_channels: 输入特征的通道数
        :param feature_channels: 中间层的特征通道数
        :param output_channels: 输出特征的通道数
        """
        super(SideViewFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, input_features, mask):
        # 应用掩码
        masked_features = (input_features * mask).contiguous()  # [B, C, H, W]，保证连续性
        # 提取特征
        x = F.relu(self.conv1(masked_features))
        x = torch.sigmoid(self.conv2(x))
        return x.contiguous()  # 保证内存连续性


class SMPLSideViewFeatureExtractor(nn.Module):
    def __init__(self, input_channels, normal_output_channels=3, feature_channels=32, output_size=(256, 256)):
        """
        用于提取 SMPL-X 左、右和后视图特征的类。
        :param input_channels: 输入特征的通道数。
        :param normal_output_channels: 法线预测网络的输出通道数（默认为3）。
        :param feature_channels: 特征提取器中间层的通道数。
        :param output_size: 输出特征的最终尺寸 (H, W)。
        """
        super(SMPLSideViewFeatureExtractor, self).__init__()
        self.output_size = output_size

        # 初始化法线预测网络
        self.normal_net = NormalPredictionNetwork(input_channels, normal_output_channels)

        # 初始化侧视图特征提取器
        self.feature_extractor = SideViewFeatureExtractor(input_channels, feature_channels, normal_output_channels)

        # 定义视角方向向量
        self.axis_vectors = {
            'T_normal_B': [0, 0, -1],  # 后视图方向
            'T_normal_R': [1, 0, 0],   # 右视图方向
            'T_normal_L': [-1, 0, 0],  # 左视图方向
        }

    def generate_view_mask(self, normals, axis_vector, img_size, device):
        """
        使用 PyTorch 张量操作生成视角掩码，支持批处理输入。
        :param normals: 法线特征图 [B, 3, H, W]（直接从网络预测）。
        :param axis_vector: 视角方向向量 [3]。
        :param img_size: 掩码尺寸 (H, W)。
        :param device: 使用的设备。
        :return: 批量掩码 [B, 1, H, W]。
        """
        # 将 axis_vector 转为张量
        axis_vector = torch.tensor(axis_vector, device=device).view(1, 3, 1, 1)  # [1, 3, 1, 1]

        # 计算法线与视角方向的点积
        dot_product = torch.sum(normals * axis_vector, dim=1, keepdim=True)  # [B, 1, H, W]

        # 使用 sigmoid 函数替代离散的阈值操作
        # 10 是放大因子，用于使 sigmoid 更接近于离散函数
        mask = torch.sigmoid(10 * (dot_product - 0.7))  # 生成平滑的掩码 [B, 1, H, W]

        # 插值掩码到指定尺寸
        mask = F.interpolate(mask, size=img_size, mode="bilinear", align_corners=False).contiguous()  # 保证连续性
        return mask

    def forward(self, input_features, img_size=(64, 64)):
        """
        从输入特征中提取 SMPL-X 左、右和后视图特征。
        :param input_features: 输入特征图 [B, C, H, W]。
        :param img_size: 掩码生成的尺寸 (H, W)。
        :return: 包含各视角特征的字典。
        """
        # 使用法线预测网络生成法线
        normals = self.normal_net(input_features).contiguous()  # [B, 3, H, W]

        # 初始化结果字典
        smpl_normal = {}

        # 逐视角生成掩码并提取特征
        for name, axis_vector in self.axis_vectors.items():
            # 生成视角掩码
            mask = self.generate_view_mask(normals, axis_vector, img_size, device=input_features.device)
            
            # 提取侧视图特征
            features = self.feature_extractor(input_features, mask)  # [B, 3, H, W]
            
            # 插值到目标尺寸
            features = F.interpolate(features, size=self.output_size, mode="bilinear", align_corners=False).contiguous()
            
            # 保存到字典
            smpl_normal[name] = features

        return smpl_normal
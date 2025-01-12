import torch
import torch.nn as nn

from .builder import LOSS

class JointsCrossEntropyDiceLoss(nn.Module):
    def __init__(self,Bce_weight,Dice_weight):
        super(JointsCrossEntropyDiceLoss, self).__init__()
        self.cross_entropy_loss = nn.BCEWithLogitsLoss()  # 使用二值交叉熵损失
        self.Bce_weight =Bce_weight
        self.Dice_weight =Dice_weight
        self.smooth = 1e-6  # 用于防止Dice损失计算中的除零错误，与eps类似

    def dice_loss(self, pred, target):
        # 使用 Sigmoid 将预测值转换为概率
        pred = torch.sigmoid(pred)

        # 检查 pred 和 target 的形状
        #print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")

        # 确保 pred 和 target 的维度是正确的 [batch_size, depth, height, width]
        intersection = (pred * target).sum(dim=(1, 2, 3))  # 调整 sum 维度索引
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

        # 使用公式一计算 Dice 损失
        dice = 2.0 * (intersection + self.smooth) / (union + 2 * self.smooth)
        return 1 - dice.mean()  # Dice 损失为 1 - Dice 系数

    def forward(self, output, target):
        num_joints = output.size(1)

        # 保持输入张量的形状，不再 reshape，因为原始维度已经是正确的形状
        heatmaps_pred = output.split(1, 1)  # 逐个关节进行处理
        heatmaps_gt = target.split(1, 1)

        total_loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)  # 去除关节的维度，使得维度为 [batch_size, depth, height, width]
            heatmap_gt = heatmaps_gt[idx].squeeze(1)

            # 计算交叉熵损失
            ce_loss = self.cross_entropy_loss(heatmap_pred, heatmap_gt)

            # 计算 Dice 损失
            dice_loss = self.dice_loss(heatmap_pred, heatmap_gt)

            # 将交叉熵和 Dice 损失结合
            combined_loss = self.Bce_weight * ce_loss + self.Dice_weight * dice_loss

            total_loss += combined_loss

        return total_loss / num_joints

def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


#新的BerHu的损失计算
class BerHu_loss(nn.Module):
    def __init__(self, c=0.2, ignore_index=255):
        super(BerHu_loss, self).__init__()
        self.c = c
        self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):
        """out and label shape both are [B, 1, H, W], float type"""
        # Generate mask to ignore pixels where label == ignore_index
        mask = (label != self.ignore_index).all(dim=1, keepdim=True)  
        
        # Select only valid (non-ignored) pixels
        masked_out = torch.masked_select(out, mask)         # Predicted values
        masked_label = torch.masked_select(label, mask)     # True values

        # Calculate the absolute difference
        diff = torch.abs(masked_out - masked_label)
        delta = self.c * torch.max(diff).item()             # delta is a scalar

        # BerHu loss calculation
        berhu_loss = torch.where(diff <= delta, diff, (diff**2 + delta**2) / (2 * delta))

        if reduction == 'mean':
            return torch.mean(berhu_loss)
        elif reduction == 'sum':
            return torch.sum(berhu_loss)
        elif reduction == 'none':
            return berhu_loss



class MultiViewLoss(nn.Module):
    def __init__(self, weights=None,c=0.2, ignore_index=255):
        super(MultiViewLoss, self).__init__()
        self.criterion = BerHu_loss(c=c, ignore_index=ignore_index)
        self.weights = weights if weights else [0.1, 0.1, 0.1]

    def forward(self, output, target):
        dec_B, dec_R, dec_L = output
        target_B, target_R, target_L = target

        loss_B = self.criterion(dec_B, target_B)
        loss_R = self.criterion(dec_R, target_R)
        loss_L = self.criterion(dec_L, target_L)

        # Weighted sum of multi-view losses
        total_loss = (self.weights[0] * loss_B +
                      self.weights[1] * loss_R +
                      self.weights[2] * loss_L)

        return total_loss  # Already averaged per batch and views
        


@LOSS.register_module
class L1LossDimSMPL(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPL, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'][:, :pred_uvd.shape[1]]
        target_uvd_weight = labels['target_weight_29'][:, :pred_uvd.shape[1]]
        loss_uvd = weighted_l1_loss(output.pred_uvd_jts, target_uvd, target_uvd_weight, self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        return loss



@LOSS.register_module
class L1LossDimSMPLCam(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']
        self.heatmap_weight = self.elements['HEATMAP_WEIGHT']
        self.view_weight =self.elements['VIEW_WEIGHT']
        self.bce_weight =self.elements['BCE_WEIGHT']
        self.dice_weight =self.elements['DICE_WEIGHT']
        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40

        #创建 JointsMSELoss 实例
        self.joints_cross_dice_loss = JointsCrossEntropyDiceLoss(self.bce_weight,self.dice_weight)


        # 创建 MultiViewLoss 实例
        self.multi_view_loss = MultiViewLoss()


    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]

        loss_uvd = weighted_l1_loss(
            pred_uvd.reshape(batch_size, -1),
            target_uvd.reshape(batch_size, -1),
            target_uvd_weight.reshape(batch_size, -1), self.size_average)
        
        #计算新的分支的关节损失
        heatmap_loss = self.joints_cross_dice_loss(output.uvd_heatmap, labels['uvd_heatmap'])

        # 计算多视图损失
        output_views = (output.initial_view_B, output.initial_view_R, output.initial_view_L)
        target_views = (labels['initial_view_B'], labels['initial_view_R'], labels['initial_view_L'])
        view_loss = self.multi_view_loss(output_views, target_views)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight
        loss += view_loss * self.view_weight  # 添加多视图损失并乘以权重因子

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight
        loss += heatmap_loss * self.heatmap_weight  # 添加新的分支的关节损失

        ###############3
        if hasattr(output, 'x_res'):
            # print("loss", loss)
            loss_mask = torch.abs(output.x_res).mean() * 2
            loss += loss_mask
            # print("loss_mask", loss_mask)
        ################

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_trans = output.cam_trans * smpl_weight
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            loss += 0.1 * (trans_loss + scale_loss)
        else:
            loss += 1 * (trans_loss + scale_loss)

        return loss

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample
import sys
# sys.path.append(r'/home/um202273190/zxy/MRNet1/MRNet/methods/MRNet')
sys.path.append(r'D:\pythonCode\conpare\reTrain\xxx1\MRNet\methods\MRNet')
import numpy as np
from mobilenetv3 import MobileNetV3LargeEncoder
from resnet import ResNet50Encoder
from lraspp import LRASPP
from decoder import RecurrentDecoder, Projection

class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class TransLayer(nn.Module):
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        self.c5_down = nn.Sequential(
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=2048, out_dim=out_c),
        )
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c5, c4, c3, c2, c1


class SIU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        s = self.conv_s_pre_up(s)
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel
def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace
    # print('***********bilateralFilter**********************')
    # print(f'batch_img{batch_img.shape}')
    pad = (ksize - 1) // 2
    # print(f'pad{pad}')
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    # print(f'patches{patches.shape}')
    patch_dim = patches.dim() # 6
    # 求出像素亮度差
    # print(f'batch_img.unsqueeze(-1).unsqueeze(-1){batch_img.unsqueeze(-1).unsqueeze(-1).shape}')
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # print(f'diff_color{diff_color.shape}')
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    # laplace = torch.tensor([[0, 1, 0],
    #                     [1, -4, 1],
    #                     [0, 1, 0]], dtype=torch.float, requires_grad=True).view(1, 1, 3, 3)
    # print(f'laplace shape{laplace.repeat(1, 3, 1, 1).shape}')
    #x = torch.from_numpy(weighted_pix.transpose([2, 0, 1])).unsqueeze(0).float()
    #y = F.conv2d(weighted_pix, laplace.repeat(1, 3, 1, 1), stride=1, padding=1,)
    #y = y.squeeze(0).numpy().transpose(1, 2, 0)

    return weighted_pix
def conv1(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# @MODELS.register()
# class MRNet(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
#         self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
#         self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])
#
#         self.d5 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)
#
#     def encoder_translayer(self, x):
#         en_feats = self.shared_encoder(x)
#         trans_feats = self.translayer(en_feats)
#         return trans_feats
#
#     def body(self, l_scale, m_scale, s_scale):
#         l_trans_feats = self.encoder_translayer(l_scale)
#         m_trans_feats = self.encoder_translayer(m_scale)
#         s_trans_feats = self.encoder_translayer(s_scale)
#
#         feats = []
#         for l, m, s, layer in zip(l_trans_feats, m_trans_feats, s_trans_feats, self.merge_layers):
#             siu_outs = layer(l=l, m=m, s=s)
#             feats.append(siu_outs)
#
#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)
#
#     def train_forward(self, data, **kwargs):
#         assert not {"image1.5", "image1.0", "image0.5", "mask"}.difference(set(data)), set(data)
#
#         output = self.body(
#             l_scale=data["image1.5"],
#             m_scale=data["image1.0"],
#             s_scale=data["image0.5"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str
#
#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             l_scale=data["image1.5"],
#             m_scale=data["image1.0"],
#             s_scale=data["image0.5"],
#         )
#         return output["seg"]
#
#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)
#
#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups
#


@MODELS.register()
class MRNet(BasicModelClass):
    def __init__(self):
        super().__init__()
        kernel = [[0, 2, 0],
                  [1, -6, 1],
                  [0, 2, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # print(f'kernel{kernel.shape}')
        kernel = np.repeat(kernel, 3, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)

        self.sobel_x = nn.Parameter(data=np.repeat(torch.FloatTensor([[-1, -1, -2, -1, -1],
                                                                      [-1, -1, -2, -1, -1],
                                                                      [0, 0, 0, 0, 0],
                                                                      [1, 1, 2, 1, 1],

                                                                      [1, 1, 2, 1, 1]]).unsqueeze(0).unsqueeze(0), 3,
                                                   axis=0), requires_grad=True)

        self.sobel_y = nn.Parameter(data=np.repeat(torch.FloatTensor([[-1, -1, 0, 1, 1],
                                                                      [-1, -1, 0, 1, 1],
                                                                      [-2, -2, 0, 2, 2],
                                                                      [-1, -1, 0, 1, 1],
                                                                      [-1, -1, 0, 1, 1]]).unsqueeze(0).unsqueeze(0), 3,
                                                   axis=0), requires_grad=True)
        self.para1 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.para2 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.para = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        self.backbone1 = MobileNetV3LargeEncoder(True)
        self.backbone2 = MobileNetV3LargeEncoder(True)
        self.aspp = LRASPP(960, 128)
        self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        self.out_layer_01 = nn.Conv2d(32, 1, 1)
        self.shallow_feat0 = nn.Sequential(conv1(3, 6, 1, bias=False),
                                           CAB(6, 3, 4, bias=False, act=torch.nn.SELU()))
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 3)
        self.detai_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.padding = torch.nn.Conv2d(3, 3, 3, padding=2)
        self.shared_encoder = timm.create_model(model_name="resnet50", in_chans=3, features_only=True)
        self.shared_encoder.load_state_dict(torch.load("D:/pythonCode/conpare/reTrain/xxx1/MRNet/pre/resnet50-0676ba61.pth"),False)
        # self.shared_encoder.load_state_dict(torch.load("/home/um202273190/zxy/MRNet1/MRNet/pre/resnet50-0676ba61.pth"),False)

        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])
        # Downloading: "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth"
        # to / home / song /.cache / torch / hub / checkpoints / resnet34 - 43635321.
        # pth

        self.d5 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)
        self.shallow_feat1 = nn.Sequential(conv1(128, 1024, 3, bias=False),
                                           CAB(1024, 3, 4, bias=False, act=torch.nn.SELU()))
    def encoder_translayer(self, x):


        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # import cv2
        # print(x[0].shape)
        # tem = x[0]
        # print(torch.permute(x[0],(1,2,0)))
        # # all = torch.zeros([f4.shape[2],f4.shape[3]])
        # # for i in range(0,f4.shape[1]):
        # channel_image = torch.permute(tem,(1,2,0)).cpu().detach().numpy().copy()
        #     # all +=channel_image
        # # channel_image -= channel_image.mean()
        # # channel_image /= channel_image.std()
        #
        # # channel_image *= 64
        # # channel_image += 128
        # # channel_image = np.clip(channel_image, 0, 255).astype("uint8")
        #
        # plt.imshow(channel_image)
        # plt.show()


        edge_t = bilateralFilter(x, 3)
        e1 = edge_t[:, 0, :, :].unsqueeze(1)
        e2 = edge_t[:, 1, :, :].unsqueeze(1)
        e3 = edge_t[:, 2, :, :].unsqueeze(1)
        edge1 = F.conv2d(e1, self.weight, padding=1)
        edge2 = F.conv2d(e2, self.weight, padding=1)
        edge3 = F.conv2d(e3, self.weight, padding=1)
        ssm = self.para1 * edge1 + self.para2 * edge2 + self.para * edge3
        tem_edge = torch.sigmoid(ssm)
        edge_t = tem_edge
        f1, f2, f3, f4 = self.backbone1(x)
        print(f'f4{f4.shape}')

        _, _, _, edge = self.backbone2(edge_t)
        print(f'edge{edge.shape}')
        # plt.imshow(edge[0][0,:,:].cpu().detach().numpy())
        # plt.show()

        # print(f'f1{f1.shape}f2{f2.shape}f3{f3.shape}f4{f4.shape}')
        f4, r4, r3, r2, r1 = self.aspp(f4, edge)
        print(f'f4{f4.shape}')
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # # print(f4[0][0,:,:].shape)
        # # all = torch.zeros([f4.shape[2],f4.shape[3]])
        # # for i in range(0,f4.shape[1]):
        # channel_image = f4[0][0,:,:].cpu().detach().numpy().copy()
        #     # all +=channel_image
        # # all -= all.mean()
        # # all /= all.std()
        #
        # # channel_image *= 64
        # # channel_image += 128
        # channel_image = np.clip(channel_image, 0, 255).astype("uint8")
        #
        # plt.imshow(channel_image)
        # plt.show()


        # print(f'after aspp  {f4.shape}')
        x1, x2,x3,x4 = self.decoder(x, f1, f2, f3, f4, r1, r2, r3, r4)
        # seg = self.project_seg(hid)
        # tem_edge = torch.sigmoid(seg)
        # print(f'en_featsf1{f1.shape}')
        # print(f'en_featsf2{f2.shape}')
        # print(f'en_featsf2{f3.shape}')
        # print(f'en_featsf4{f4.shape}')
        # f4_n =self.shallow_feat1(f4)
        # new_input = x + x*seg

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # import cv2
        
        # all = torch.zeros([x1.shape[2],x1.shape[3]])
        # for i in range(0,x1.shape[1]):
        #     # channel_image = x1[0][0,:,:].cpu().detach().numpy().copy()
        #     # all +=channel_image
        #     channel_image = x1[0][i,:,:].cpu().detach().numpy().copy()
        #     all +=channel_image
        #     plt.imshow(all.unsqueeze(-1))
        #     plt.show()

        en_feats = self.shared_encoder(x)





        if x1.shape[-1]==288:
            # print(f'ccccccccccccccccccccccccccccccccccccs')
            tem_0 = en_feats[0][:,:x1.shape[1],:,:]

            tem_1 = en_feats[1][:,:x2.shape[1],:,:]
            tem_2 = en_feats[2][:,:x3.shape[1],:,:]
            tem_3 = en_feats[3][:,:x4.shape[1],:,:]
            # print(f'x1{x1.shape}')
            a = tem_0.clone() + x1
            b = tem_1.clone() + x2
            c = tem_2.clone() + x3
            d = tem_3.clone() + x4

            # print(torch.cat((a,en_feats[0][:,x1.shape[1]:,:,:]),1).shape)
            en_feats1= [torch.cat((a,en_feats[0][:,x1.shape[1]:,:,:]),1),torch.cat((b,en_feats[1][:,x2.shape[1]:,:,:]),1),torch.cat((c,en_feats[2][:,x3.shape[1]:,:,:]),1),torch.cat((d,en_feats[3][:,x4.shape[1]:,:,:]),1),en_feats[4]]
            # en_feats[1][:,:x2.shape[1],:,:] = tem_1 +x2
            # en_feats[2][:,:x3.shape[1],:,:] = tem_2+x3
            # en_feats[3][:,:x4.shape[1],:,:] = tem_3+x4
        # print(f'en_feats0{en_feats[0].shape}')
        # print(f'en_feats1{en_feats[1].shape}')
        # print(f'en_feats2{en_feats[2].shape}')
        # print(f'en_feats3{en_feats[3].shape}')
        # print(f'en_feats4{en_feats[4].shape}')
        else:
            en_feats1 = en_feats
        trans_feats = self.translayer(en_feats1)
        return trans_feats


    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = self.encoder_translayer(l_scale)
        m_trans_feats = self.encoder_translayer(m_scale)
        s_trans_feats = self.encoder_translayer(s_scale)



        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # import cv2
        x11 =m_trans_feats[0]
        for i in range(0,x11.shape[1]):
            channel_image = x11[0][i,:,:].unsqueeze(-1).cpu().detach().numpy().copy()
                # all +=channel_image
            # plt.imshow(channel_image)
            # plt.show()

        # print(f'Upsampling {l_trans_feats}')

        feats = []
        for l, m, s, layer in zip(l_trans_feats, m_trans_feats, s_trans_feats, self.merge_layers):
            siu_outs = layer(l=l, m=m, s=s)
            feats.append(siu_outs)





        x = self.d5(feats[0])
        x = cus_sample(x, mode="scale", factors=2)

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # import cv2
        # x11 =x
        # for i in range(0,x11.shape[1]):
        #     channel_image = x11[0][i,:,:].unsqueeze(-1).cpu().detach().numpy().copy()
        #         # all +=channel_image
        #     plt.imshow(channel_image)
        #     plt.show()
        # print(x.shape)
        x = self.d4(x + feats[1])
        x = cus_sample(x, mode="scale", factors=2)
        # print(f'cuds_sample1x{x.shape}')
        x = self.d3(x + feats[2])
        x = cus_sample(x, mode="scale", factors=2)
        # print(f'cuds_sample1x{x.shape}')
        x = self.d2(x + feats[3])
        x = cus_sample(x, mode="scale", factors=2)

        # print(f'cuds_sample1x{x.shape}')
        # print(x.shape)


        x = self.d1(x + feats[4])
        x = cus_sample(x, mode="scale", factors=2)

        # print(x.shape)

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # import cv2
        # x11 =x
        # # all = torch.zeros([x11.shape[2],x1.shape[3]])
        # for i in range(0,x11.shape[1]):
        #     # channel_image = x1[0][0,:,:].cpu().detach().numpy().copy()
        #     # all +=channel_image
        #     channel_image = x11[0][i,:,:].unsqueeze(-1).cpu().detach().numpy().copy()
        #         # all +=channel_image
        #     plt.imshow(channel_image)
        #     plt.show()

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # import cv2
        # for i in range(0,x.shape[1]):
        #     channel_image = x[0][i,:,:].unsqueeze(-1).cpu().detach().numpy().copy()
            
        #     plt.imshow(channel_image)
        #     plt.show()
        logits = self.out_layer_01(self.out_layer_00(x))
        return dict(seg=logits)

    def train_forward(self, data, **kwargs):
        assert not {"image1.5", "image1.0", "image0.5", "mask"}.difference(set(data)), set(data)

        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
        )
        return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups

@MODELS.register()
class MRNet_CK(MRNet):
    def __init__(self):
        super().__init__()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def encoder(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x0, x1, x2, x3, x4 = self.shared_encoder(x)
        return x0, x1, x2, x3, x4

    def trans(self, x0, x1, x2, x3, x4):
        x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])
        return x5, x4, x3, x2, x1

    def decoder(self, x5, x4, x3, x2, x1):
        x = self.d5(x5)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + x4)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + x3)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + x2)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + x1)
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return logits

    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
        m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
        s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)
        l_trans_feats = checkpoint(self.trans, *l_trans_feats)
        m_trans_feats = checkpoint(self.trans, *m_trans_feats)
        s_trans_feats = checkpoint(self.trans, *s_trans_feats)

        feats = []
        for layer_idx, (l, m, s) in enumerate(zip(l_trans_feats, m_trans_feats, s_trans_feats)):
            siu_outs = checkpoint(self.merge_layers[layer_idx], l, m, s)
            feats.append(siu_outs)

        logits = checkpoint(self.decoder, *feats)
        return dict(seg=logits)

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
import sys
# sys.path.append('/mnt/data/zhangxinyue/NAS/compareCODCode/code/CAM-Seg/model')
sys.path.append(r'D:\pythonCode\conpare\reTrain\MRNet\MRNet\methods\MRNet')
import numpy as np
from mobilenetv3 import MobileNetV3LargeEncoder
from resnet import ResNet50Encoder
from lraspp import LRASPP
from decoder import RecurrentDecoder, Projection
from fast_guided_filter import FastGuidedFilterRefiner
from deep_guided_filter import DeepGuidedFilterRefiner
from utils.builder import MODELS
from methods.module.base_model import BasicModelClass
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

@MODELS.register()
class MattingNetwork(BasicModelClass):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        # print('init the MattingNetwork')
        kernel = [[0, 2, 0],
                        [1, -6, 1],
                        [0, 2, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # print(f'kernel{kernel.shape}')
        kernel = np.repeat(kernel, 3, axis = 0)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)


        self.sobel_x = nn.Parameter(data =np.repeat(torch.FloatTensor([[-1,-1, -2,-1, -1],
                        [-1,-1, -2,-1, -1],
                        [0, 0, 0,0,0],
                        [1,1, 2,1, 1],

                        [1,1, 2,1, 1]]).unsqueeze(0).unsqueeze(0), 3, axis = 0), requires_grad=True)

        self.sobel_y = nn.Parameter(data =np.repeat(torch.FloatTensor([[-1,-1, 0,1, 1],
                                [-1,-1, 0,1, 1],
                                [-2, -2, 0, 2,2],
                                [-1,-1, 0,1, 1],
                                [-1,-1, 0,1, 1]]).unsqueeze(0).unsqueeze(0), 3, axis = 0), requires_grad=True)
        self.para1 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.para2 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.para = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        # self.sobel_x = nn.Parameter(data =np.repeat(torch.FloatTensor([[-1, -2, -1],
        #
        #                 [0,  0,0],
        #                 [1, 2, 1]]).unsqueeze(0).unsqueeze(0), 3, axis = 0), requires_grad=False)
        #
        # self.sobel_y = nn.Parameter(data =np.repeat(torch.FloatTensor([
        #                         [-1, 0, 1],
        #                         [-2,  0, 2],
        #                         [-1, 0, 1]]).unsqueeze(0).unsqueeze(0), 3, axis = 0), requires_grad=False)


        # print(f'self.sobel_y{self.sobel_y.shape}')

        # print(f'self.weight{self.weight.shape}')
        if variant == 'mobilenetv3':
            self.backbone1 = MobileNetV3LargeEncoder(True)
            self.backbone2 = MobileNetV3LargeEncoder( True)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])

        self.shallow_feat0 = nn.Sequential(conv1(3, 6, 1, bias=False),
                                           CAB(6, 3, 4, bias=False, act=torch.nn.SELU()))
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 3)
        self.detai_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.padding = torch.nn.Conv2d(3,3,3,padding=2)
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        edge_t = bilateralFilter(src_sm,3)
        e1 = edge_t[:,0,:,:].unsqueeze(1)
        e2 = edge_t[:,1,:,:].unsqueeze(1)
        e3 = edge_t[:,2,:,:].unsqueeze(1)
        edge1 = F.conv2d(e1, self.weight, padding=1)
        edge2 = F.conv2d(e2, self.weight, padding=1)
        edge3 = F.conv2d(e3, self.weight, padding=1)

        # e1 = src_sm[:,0,:,:].unsqueeze(1)
        # e2 = src_sm[:,1,:,:].unsqueeze(1)
        # e3 = src_sm[:,2,:,:].unsqueeze(1)
        # edge1 = F.conv2d(e1, self.weight, padding=0)
        # edge2 = F.conv2d(e2, self.weight, padding=0)
        # edge3 = F.conv2d(e3, self.weight, padding=0)


        # src_sm_1 = bilateralFilter(src_sm,3)
        # s1_1 = torch.abs(F.conv2d(src_sm_1, self.sobel_x, padding = 2))
        # s1_2 = torch.abs( F.conv2d(src_sm_1, self.sobel_y, padding = 2))
        # s1_3 = torch.abs(F.conv2d(src_sm, self.weight,padding = 1))
        # print(edge1.shape,edge2.shape, edge3.shape)
        ssm = self.para1*edge1 + self.para2*edge2 +self.para*edge3

        # print(ssm.shape)
        # newi = torch.sigmoid(ssm)*src_sm
        # import matplotlib.pyplot as plt
        # plt.imshow(ssm[0,:,:,:].cpu().permute(1,2,0).detach().numpy())
        # plt.show()
        # plt.imshow(newi[0,:,:,:].cpu().permute(1,2,0).detach().numpy())
        # plt.show()
        # print(newi.shape)

        # edge_cat = torch.cat((edge1,edge2,edge3),1)
        tem_edge =torch.sigmoid(ssm)
        edge_t =tem_edge
        # print(f'tem_edge{tem_edge.shape}')


        # print(edge_cat.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(edge_cat[0,:,:,:].cpu().permute(1,2,0).detach().numpy())
        # plt.show()
        # import matplotlib.pyplot as plt
        # plt.imshow(edge_t[0,:,:,:].cpu().permute(1,2,0).detach().numpy())
        # plt.show()
        # e1 = edge_t[:,0,:,:].unsqueeze(1)
        # e2 = edge_t[:,1,:,:].unsqueeze(1)
        # e3 = edge_t[:,2,:,:].unsqueeze(1)
        # src_sm_d = self.detai_conv(src_sm)

        # H = src_sm_d.size(2)
        # W = src_sm_d.size(3)
        #
        # # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        #
        # # Two Patches for Stage 2
        # x2top_img = src_sm_d[:, :, 0:int(H / 2), :]
        # x2bot_img = src_sm_d[:, :, int(H / 2):H, :]
        # # print("********************")
        # # Four Patches for Stage 1
        # x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        # x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        # x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        # x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]
        # # Eight Patches for Stage 0
        # x0ltop_img1 = x1ltop_img[:, :, :, 0:int(W / 4)]
        # x0ltop_img2 = x1ltop_img[:, :, :, int(W / 4):int(W / 2)]
        # x0rtop_img1 = x1rtop_img[:, :, :, 0:int(W / 4)]
        # x0rtop_img2 = x1rtop_img[:, :, :, int(W / 4):int(W / 2)]
        # x0lbot_img1 = x1lbot_img[:, :, :, 0:int(W / 4)]
        # x0lbot_img2 = x1lbot_img[:, :, :, int(W / 4):int(W / 2)]
        # x0rbot_img1 = x1rbot_img[:, :, :, 0:int(W / 4)]
        # x0rbot_img2 = x1rbot_img[:, :, :, int(W / 4):int(W / 2)]
        # ##-------------------------------------------
        # ##-------------- Stage 0---------------------
        # ##-------------------------------------------
        # x0ltop_img1 = self.shallow_feat0(x0ltop_img1)
        # # print(" aaaaaaaaaaax0ltop_img1 shape{}".format(x0ltop_img1.size()))
        # x0ltop_img2 = self.shallow_feat0(x0ltop_img2)
        # # print(" aaaaaaaaaaaaax0ltop_img2 shape{}".format(x0ltop_img2.size()))
        # x0rtop_img1 = self.shallow_feat0(x0rtop_img1)
        # x0rtop_img2 = self.shallow_feat0(x0rtop_img2)
        # x0lbot_img1 = self.shallow_feat0(x0lbot_img1)
        # x0lbot_img2 = self.shallow_feat0(x0lbot_img2)
        # x0rbot_img1 = self.shallow_feat0(x0rbot_img1)
        # x0rbot_img2 = self.shallow_feat0(x0rbot_img2)
        #
        # # print(f'x0ltop_img1{x0ltop_img1.shape}')
        # ## Concat deep features--->4
        # feat0_1 = torch.cat((x0ltop_img1, x0ltop_img2),
        #                     3)  # [torch.cat((k, v), 2) for k, v in zip(x0ltop_img1, x0ltop_img2)]
        # feat0_2 = torch.cat((x0rtop_img1, x0rtop_img2),
        #                     3)  # [torch.cat((k, v), 2) for k, v in zip(x0rtop_img1, x0rtop_img2)]
        # feat0_3 = torch.cat((x0lbot_img1, x0lbot_img2),
        #                     3)  # [torch.cat((k, v), 2) for k, v in zip(x0lbot_img1, x0lbot_img2)]
        # feat0_4 = torch.cat((x0rbot_img1, x0rbot_img2),
        #                     3)  # [torch.cat((k, v), 2) for k, v in zip(x0rbot_img1, x0rbot_img2)]
        #
        # ## Concat deep features--->2
        # feat0_1_0 = torch.cat((feat0_1, feat0_2), 3)  # [torch.cat((k, v), 2) for k, v in zip(feat0_1, feat0_2)]
        # # print("concat feat 0_1 {}".format(feat0_1_0.size()))
        # feat0_2_1 = torch.cat((feat0_3, feat0_4), 3)  # [torch.cat((k, v), 2) for k, v in zip(feat0_3, feat0_4)]
        # ## Concat deep features--->1
        # feat0_1_0_0 = torch.cat((feat0_1_0, feat0_2_1),
        #                         3)  # [torch.cat((k, v), 2) for k, v in zip(feat0_1_0, feat0_2_1)]
        # # print(f'feat0_1_0_0{feat0_1_0_0[:,:3,:,:].shape}')
        f1, f2, f3, f4 = self.backbone1(src_sm)
        # edge_cat = torch.cat((edge1,edge2,edge3),1)
        _,_,_,edge = self.backbone2(edge_t)
        # print(f'x0ltop_img1{x0ltop_img1.shape}feat0_1_0_0{feat0_1_0_0.shape} f4{f4.shape}')
        f4, r4 ,r3, r2,r1= self.aspp(f4,edge)
        # print(r4.shape, r3.shape, r2.shape, r1.shape)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        # if not segmentation_pass:
        #     fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
        #     if downsample_ratio != 1:
        #         fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
        #     fgr = fgr_residual + src
        #     fgr = fgr.clamp(0., 1.)
        #     pha = pha.clamp(0., 1.)
        #     return [fgr, pha, *rec]
        # else:
        seg = self.project_seg(hid)
        #print(seg.shape)

        # return seg, tem_edge.repeat(1,3,1,1)
        return seg, tem_edge

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
if __name__=="__main__":
    print('model test')
    a = MattingNetwork()
    src = torch.rand((1,3,512,512))
    #r1 = torch.rand((1,1,512,512))#
    a(src)

    from thop import profile

    macs, params = profile(a, inputs=(src, ))

    print("MACs=", str(macs / 1e9) + '{}'.format("G"))
    print("MACs=", str(macs / 1e6) + '{}'.format("M"))

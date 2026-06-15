import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter

from sample4geo.Utils import init

class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, x_out21


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='kaiming'): # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class GlobalGuidedEnhancementFlow(nn.Module):
    def __init__(self, channels, reduction=4):
        super(GlobalGuidedEnhancementFlow, self).__init__()
        hidden = max(channels // reduction, 64)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class LocalReconstructionEnhancementFlow(nn.Module):
    def __init__(self, channels):
        super(LocalReconstructionEnhancementFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            return self.net(x.float()).to(dtype=x.dtype)


class GeometricSpatialAlignmentModule(nn.Module):
    def __init__(self, channels):
        super(GeometricSpatialAlignmentModule, self).__init__()
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 6),
        )
        nn.init.zeros_(self.localization[-1].weight)
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.localization(x).view(-1, 2, 3).to(dtype=x.dtype, device=x.device)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class SemanticGuidedViewRectificationModule(nn.Module):
    def __init__(self, channels, residual_scale=0.1):
        super(SemanticGuidedViewRectificationModule, self).__init__()
        self.global_flow = GlobalGuidedEnhancementFlow(channels)
        self.local_flow = LocalReconstructionEnhancementFlow(channels)
        self.alignment = GeometricSpatialAlignmentModule(channels)
        self.gate = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        guided = self.global_flow(x) + self.local_flow(x)
        rectified = self.alignment(guided)
        return x + self.gate.tanh() * rectified


class MultiScalePerceptionModule(nn.Module):
    def __init__(self, channels):
        super(MultiScalePerceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=5, dilation=5, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1))


class SpatialCoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super(SpatialCoordinateAttention, self).__init__()
        hidden = max(32, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.attn_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.attn_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.size()
        feat_h = F.adaptive_avg_pool2d(x, (h, 1))
        feat_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        feat = self.shared(torch.cat([feat_h, feat_w], dim=2))
        feat_h, feat_w = torch.split(feat, [h, w], dim=2)
        feat_w = feat_w.transpose(2, 3)
        return x * torch.sigmoid(self.attn_h(feat_h)) * torch.sigmoid(self.attn_w(feat_w))


class ScaleAdaptiveFeatureCalibrationModule(nn.Module):
    def __init__(self, channels, residual_scale=0.1):
        super(ScaleAdaptiveFeatureCalibrationModule, self).__init__()
        self.multiscale = MultiScalePerceptionModule(channels)
        self.coordinate_attention = SpatialCoordinateAttention(channels)
        self.gate = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        calibrated = self.coordinate_attention(self.multiscale(x))
        return x + self.gate.tanh() * calibrated


class CrossViewLocalInteractionModule(nn.Module):
    def __init__(self, channels):
        super(CrossViewLocalInteractionModule, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1, bias=False)
        self.key = nn.Conv2d(channels, channels // 8, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x, context=None):
        if context is None:
            context = x
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(context).flatten(2)
        v = self.value(context).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) / max(k.size(1) ** 0.5, 1.0), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h, w)
        return self.proj(out)


class CrossViewGlobalInteractionModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(CrossViewGlobalInteractionModule, self).__init__()
        hidden = max(channels // reduction, 64)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        return self.proj(x * self.gate(x))


class CrossViewSemanticAlignmentModule(nn.Module):
    def __init__(self, channels, residual_scale=0.1):
        super(CrossViewSemanticAlignmentModule, self).__init__()
        self.local_interaction = CrossViewLocalInteractionModule(channels)
        self.global_interaction = CrossViewGlobalInteractionModule(channels)
        self.gate = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x, context=None):
        local = self.local_interaction(x, context)
        global_feature = self.global_interaction(x + local)
        return x + self.gate.tanh() * global_feature


class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block
        self.tri_layer = TripletAttention()
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

        # define for Domain Space Alignment Module
        in_channels = 1024
        hid_channels = 2048
        out_channels = 256
        norm_layer = None
        num_layers = 2
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj.init_weights()
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj.init_weights()
        self.scale = 1.
        self.l2_norm = True
        self.num_heads = 8
        self.use_vrm = True
        self.use_sam = True
        self.use_csm = True
        self.vrm = SemanticGuidedViewRectificationModule(self.in_planes, residual_scale=0.1)
        self.sam = ScaleAdaptiveFeatureCalibrationModule(self.in_planes, residual_scale=0.1)
        self.csm = CrossViewSemanticAlignmentModule(self.in_planes, residual_scale=0.1)

    def configure_vcsa_modules(self, use_vrm=True, use_sam=True, use_csm=True, residual_scale=0.1):
        self.use_vrm = use_vrm
        self.use_sam = use_sam
        self.use_csm = use_csm
        for module in (self.vrm, self.sam, self.csm):
            module.gate.data.fill_(float(residual_scale))

    def enhanced_part_features(self, part_features, paired_part_features=None):
        original_dtype = part_features.dtype
        autocast_enabled = part_features.is_cuda
        with torch.cuda.amp.autocast(enabled=not autocast_enabled):
            x = part_features.float() if autocast_enabled else part_features
            if self.use_vrm:
                x = self.vrm(x)
            if self.use_sam:
                x = self.sam(x)
            context = paired_part_features
            if context is not None:
                context = context.float() if autocast_enabled else context
                if self.use_vrm:
                    context = self.vrm(context)
                if self.use_sam:
                    context = self.sam(context)
            if self.use_csm:
                x = self.csm(x, context)
        return x.to(dtype=original_dtype)

    def forward_features(self, x, paired_part_features=None):
        gap_feature, part_features = self.convnext(x)
        part_features = self.enhanced_part_features(part_features, paired_part_features).contiguous()
        return gap_feature, part_features

    def forward_pair(self, x1, x2):
        gap1, part1 = self.convnext(x1)
        gap2, part2 = self.convnext(x2)
        return self.forward_from_features(gap1, part1, paired_part_features=part2), self.forward_from_features(gap2, part2, paired_part_features=part1)

    def forward_from_features(self, gap_feature, part_features, paired_part_features=None):
        part_features = self.enhanced_part_features(part_features, paired_part_features).contiguous()
        return self.forward_head(gap_feature, part_features)

    def forward(self, x, paired_part_features=None):
        # -- backbone feature extractor
        gap_feature, part_features = self.forward_features(x, paired_part_features=paired_part_features)
        return self.forward_head(gap_feature, part_features)

    def forward_head(self, gap_feature, part_features):
        # -- Training
        if self.training:

            # 1. Domain Space Alignment Module
            b, c, h, w = part_features.shape
            pfeat = part_features.contiguous().flatten(2)  # (bs, c, h*w)
            W = self.proj(pfeat).contiguous()  # transfer 1024 to 256

            W = F.normalize(W, dim=1).contiguous() if self.l2_norm else W
            W = (W * (1 / self.scale)).contiguous()
            W = F.softmax(W, dim=2).contiguous()

            pfeat_align = torch.cat([pfeat, W], dim=1)
            # pfeat_align = pfeat

            # 2. triplet attention
            tri_features = self.tri_layer(part_features)  # 旋转另外两个轴，返回两个一样的特征体(bs, 1024, 12, 12); (bs, 1024, 12, 12)
            convnext_feature = self.classifier1(gap_feature)  # class: (bs, 701); feature: (bs, 512)
            tri_list = []
            for i in range(self.block):
                tri_list.append(tri_features[i].mean([-2, -1]))  # average pooling, 一张图变一个像素
            triatten_features = torch.stack(tri_list, dim=2)  # 把另外两个轴旋转的特征体
            if self.block == 0:
                y = []
            else:
                y = self.part_classifier(self.block, triatten_features,
                                         cls_name='classifier_mcb')  # 把另外两个轴旋转的feature也做分类
            y = y + [convnext_feature]  # 三个分支连起来
            if self.return_f:  # return_f是triplet loss的设置，0.3
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return pfeat_align, cls, features, gap_feature, part_features

        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass

        return gap_feature, part_features

    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

    def fine_grained_transform(self):

        pass


def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class, block=block, return_f=return_f, resnet=resnet)
    return model

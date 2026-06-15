import torch.nn as nn
from .ConvNext import make_FCFE_model


def _split_rgb_normal(x):
    if isinstance(x, (tuple, list)):
        return x[0], x[1]
    return x, None


class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, M=32, return_f=False, resnet=False, use_fcfe=False,
                 use_normals=False, fcfe_dual_backbone=False, pretrained=True):
        super(two_view_net, self).__init__()
        self.model_1 = make_FCFE_model(num_class=class_num, block=block, M=M, return_f=return_f, resnet=resnet,
                                      use_fcfe=use_fcfe, use_normals=use_normals,
                                      fcfe_dual_backbone=fcfe_dual_backbone, pretrained=pretrained)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            rgb1, normal1 = _split_rgb_normal(x1)
            y1 = self.model_1(rgb1, normal1)

        if x2 is None:
            y2 = None
        else:
            rgb2, normal2 = _split_rgb_normal(x2)
            y2 = self.model_1(rgb2, normal2)
        return y1, y2


class three_view_net(nn.Module):
    def __init__(self, class_num, share_weight=False, block=4, M=32, return_f=False, resnet=False, use_fcfe=False,
                 use_normals=False, fcfe_dual_backbone=False, pretrained=True):
        super(three_view_net, self).__init__()
        self.share_weight = share_weight
        self.model_1 = make_FCFE_model(num_class=class_num, block=block, M=M, return_f=return_f, resnet=resnet,
                                      use_fcfe=use_fcfe, use_normals=use_normals,
                                      fcfe_dual_backbone=fcfe_dual_backbone, pretrained=pretrained)

        if self.share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = make_FCFE_model(num_class=class_num, block=block, M=M, return_f=return_f, resnet=resnet,
                                          use_fcfe=use_fcfe, use_normals=use_normals,
                                          fcfe_dual_backbone=fcfe_dual_backbone, pretrained=pretrained)

    def forward(self, x1, x2, x3, x4=None):  # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            rgb1, normal1 = _split_rgb_normal(x1)
            y1 = self.model_1(rgb1, normal1)

        if x2 is None:
            y2 = None
        else:
            rgb2, normal2 = _split_rgb_normal(x2)
            y2 = self.model_2(rgb2, normal2)

        if x3 is None:
            y3 = None
        else:
            rgb3, normal3 = _split_rgb_normal(x3)
            y3 = self.model_1(rgb3, normal3)

        if x4 is None:
            return y1, y2, y3
        else:
            rgb4, normal4 = _split_rgb_normal(x4)
            y4 = self.model_2(rgb4, normal4)
        return y1, y2, y3, y4


def make_model(opt):
    use_fcfe = getattr(opt, 'use_fcfe', False)
    use_normals = getattr(opt, 'use_normals', False)
    fcfe_dual_backbone = getattr(opt, 'fcfe_dual_backbone', False)
    pretrained = getattr(opt, 'pretrained', True)
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block, M=opt.M, return_f=opt.triplet_loss, resnet=opt.resnet,
                             use_fcfe=use_fcfe, use_normals=use_normals,
                             fcfe_dual_backbone=fcfe_dual_backbone, pretrained=pretrained)
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, share_weight=opt.share, block=opt.block, M=opt.M,
                               return_f=opt.triplet_loss, resnet=opt.resnet, use_fcfe=use_fcfe,
                               use_normals=use_normals, fcfe_dual_backbone=fcfe_dual_backbone,
                               pretrained=pretrained)
    return model

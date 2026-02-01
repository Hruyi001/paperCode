import torch.nn as nn
from .model import Safe_Net_model

class Safe_Net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, imgsize=256, save_intermediate=False):
        super(Safe_Net, self).__init__()
        self.loc_model = Safe_Net_model(num_classes=class_num, block=block, return_f=return_f, imgsize=imgsize, save_intermediate=save_intermediate)
        self.aligned_feature_map = None
        self.intermediate_outputs = {}
    # def forward(self, x1, x2):
    #     if x1 is None:
    #         y1 = None
    #     else:
    #         y1 = self.loc_model(x1)

    #     if x2 is None:
    #         y2 = None
    #     else:
    #         y2 = self.loc_model(x2)

    #     return y1, y2
    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.loc_model(x1)
            # 将 loc_model 中生成的特征图赋值给当前类的 aligned_feature_map
            self.aligned_feature_map = self.loc_model.aligned_feature_map
            # 保存中间输出
            if hasattr(self.loc_model, 'intermediate_outputs'):
                self.intermediate_outputs = self.loc_model.intermediate_outputs

        if x2 is None:
            y2 = None
        else:
            y2 = self.loc_model(x2)
            # 若处理 x2，同样更新特征图（根据需求选择是否覆盖）
            self.aligned_feature_map = self.loc_model.aligned_feature_map
            # 保存中间输出
            if hasattr(self.loc_model, 'intermediate_outputs'):
                self.intermediate_outputs = self.loc_model.intermediate_outputs

        return y1, y2

def create_model(opt):
    model_path = "./models/pretrain_backbone/vit_small_p16_224-15ec54c9.pth"
    
    # 检查是否有save_intermediate参数
    save_intermediate = getattr(opt, 'save_intermediate', False)

    # create Safe-Net
    model = Safe_Net(class_num=opt.nclasses, block=opt.block, return_f=opt.triplet_loss, imgsize=opt.h, save_intermediate=save_intermediate)
    # load pretrain param
    model.loc_model.transformer.load_param(model_path)

    return model

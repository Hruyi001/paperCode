import os
import torch
import yaml
import torch.nn as nn
from models.model import two_view_net, three_view_net


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


######################################################################
# Save model
# ---------------------------
def save_network(network, dirname, epoch_label, base_dir='./model'):
    save_dir = os.path.join(base_dir, dirname)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(base_dir, dirname, save_filename)
    model_to_save = network.module if hasattr(network, 'module') else network
    torch.save(model_to_save.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
# ---------------------------
def load_network(name, opt):
    # Load config
    base_dir = getattr(opt, 'save_dir', './model')
    dirname = os.path.join(base_dir, name)
    requested_epoch = getattr(opt, 'which_epoch', 'last')
    if requested_epoch == 'last':
        last_model_name = os.path.basename(get_model_list(dirname, 'net'))
        epoch = last_model_name.split('_')[1]
        epoch = epoch.split('.')[0]
    else:
        epoch = str(requested_epoch)
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # opt.name = config['name']
    # opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.fp16 = config['fp16']
    opt.views = config['views']
    opt.block = config['block']
    opt.M = config['M']
    opt.resnet = False
    if 'resnet' in config:
        opt.resnet = config['resnet']
    opt.use_fcfe = config.get('use_fcfe', False)
    opt.use_normals = config.get('use_normals', False)
    opt.normal_dir = config.get('normal_dir', '')
    opt.auto_generate_normals = config.get('auto_generate_normals', True)
    opt.omnidata_repo = config.get('omnidata_repo', '/root/code/omnidata_models')
    opt.omnidata_root = config.get('omnidata_root', '/root/code/omnidata_models/pretrained_models')
    opt.fcfe_dual_backbone = config.get('fcfe_dual_backbone', False)
    opt.pretrained = config.get('pretrained', True)

    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block, M=opt.M, resnet=opt.resnet,
                             use_fcfe=opt.use_fcfe, use_normals=opt.use_normals,
                             fcfe_dual_backbone=opt.fcfe_dual_backbone, pretrained=opt.pretrained)
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, share_weight=opt.share, block=opt.block, M=opt.M,
                               resnet=opt.resnet, use_fcfe=opt.use_fcfe, use_normals=opt.use_normals,
                               fcfe_dual_backbone=opt.fcfe_dual_backbone, pretrained=opt.pretrained)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch

    save_path = os.path.join(base_dir, name, save_filename)
    print('Load the model from %s' % save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network, opt, epoch


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toogle_grad(model_src, True)

from networks.unet import UNet
from networks.VNet import VNet

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "VNet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net

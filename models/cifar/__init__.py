import os
import torch
from .resnet import (
    resnet56,
    resnet110
)
from .wrn import wrn_40_2


cifar100_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../download_ckpts/cifar_teachers/"
)

archs = {
    'resnet56': resnet56,
    'resnet110': resnet110,
    'wrn_40_2': wrn_40_2
}


cifar_model_dict = {
    # teachers
    "resnet56": (
        resnet56,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet110": (
        resnet110,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.pth",
    ),
    "wrn_40_2": (
        wrn_40_2,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    )
}

def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
def get_model(model_name: str, pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    net, checkpoint = cifar_model_dict[model_name]
    model = net(num_classes=100)
    model.load_state_dict(load_checkpoint(checkpoint)['model'])
    return model




import torch
import torch.nn as nn
import torchvision.models as models

from .cricavpr.backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

vit_dict = {
    "vits": vit_small,
    "vitb": vit_base,
    "vitl": vit_large,
    "vitg": vit_giant2
}

AVAILABLE_MODELS = [
    'dinov2_vits14',
    'dinov2_vitb14',
    'dinov2_vitl14',
    'dinov2_vitg14'
]


def load_basic_model(model_name, model_pretrained=True):
    if model_name == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=model_pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if model_pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

        encoder = nn.Sequential(*layers)

    elif model_name == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=model_pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if model_pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)

    elif model_name == 'mobilenetv2':
        encoder_dim = 320
        encoder = models.mobilenet_v2(pretrained=model_pretrained)
        # capture only features and remove last several layers: ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        layers = list(encoder.features.children())[:-1]
        # layers.append(nn.Conv2d(encoder_dim, encoder_dim, kernel_size=(1, 1), bias=True))
        if model_pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                # for l in layers:
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)

    elif model_name == 'resnet18':
        encoder_dim = 512
        encoder = models.resnet18(pretrained=model_pretrained)
        # capture only features and remove last several layers: ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        layers = list(encoder.children())[:-2]
        # layers.append(nn.Conv2d(encoder_dim, encoder_dim, kernel_size=(1, 1), bias=True))
        # layers.append(list(encoder.children())[-3][0])
        if model_pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                # for l in layers:
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)

    elif model_name == 'resnet18p365':
        encoder_dim = 512
        encoder = models.resnet18(num_classes=365)
        # capture only features and remove last several layers: ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        # load the pre-trained weights
        model_file = '../pretrained_model/resnet18_places365.pth.tar'
        checkpoint = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k,
                      v in checkpoint['state_dict'].items()}
        encoder.load_state_dict(state_dict)

        layers = list(encoder.children())[:-2]
        # layers.append(nn.Conv2d(encoder_dim, encoder_dim, kernel_size=(1, 1), bias=True))
        # layers.append(list(encoder.children())[-3][0])
        if model_pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                # for l in layers:
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)

    elif model_name == 'alexnetp365':
        encoder_dim = 256
        encoder = models.alexnet(num_classes=365)
        # capture only features and remove last several layers: ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        # load the pre-trained weights
        model_file = '../pretrained_model/alexnet_places365.pth.tar'
        checkpoint = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k,
                      v in checkpoint['state_dict'].items()}
        encoder.load_state_dict(state_dict)

        layers = list(encoder.features.children())[:-2]

        if model_pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)

    elif model_name.startswith('dinov2'):
        backbone_type = model_name.split('_')[1] if len(
            model_name.split('_')) > 1 else 'vitb'
        encoder = vit_dict[backbone_type](
            patch_size=14, img_size=518, init_values=1, block_chunks=0)
        encoder_dim = encoder.embed_dim

    elif model_name.startswith('puredinov2'):
        encoder = get_pure_dinov2(num_unfrozen_blocks=2)
        encoder_dim = encoder.embed_dim
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")

    return encoder, encoder_dim


def get_pure_dinov2(backbone_name="dinov2_vitb14", num_unfrozen_blocks=2):
    assert backbone_name in AVAILABLE_MODELS, f"Backbone {backbone_name} is not recognized! Supported backbones are: {AVAILABLE_MODELS}"
    dino = torch.hub.load('facebookresearch/dinov2', backbone_name)

    dino.patch_embed.requires_grad_(False)
    dino.pos_embed.requires_grad_(False)

    for i in range(len(dino.blocks) - num_unfrozen_blocks):
        dino.blocks[i].requires_grad_(False)

    return dino

from src_ import  models
import json
import re


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_model(model_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_args = Namespace(**config)

    # AuM block type
    bimamba_type = {
        'Fo-Fo': 'none', 
        'Fo-Bi': 'v1', 
        'Bi-Bi': 'v2'
    }.get(
        model_args.aum_variant, 
        None
    )

    AuM = models.AudioMamba(
        spectrogram_size=(data_args.num_mel_bins, data_args.target_length),
        patch_size=(16, 16),
        strides=(16, 16),
        embed_dim=768,
        num_classes=model_args.n_classes,
        imagenet_pretrain=model_args.imagenet_pretrain,
        imagenet_pretrain_path=model_args.imagenet_pretrain_path,
        aum_pretrain=model_args.aum_pretrain,
        aum_pretrain_path=model_args.aum_pretrain_path,
        bimamba_type=bimamba_type,
    )

    AuM.to(model_args.device)
    AuM.eval()

    return AuM
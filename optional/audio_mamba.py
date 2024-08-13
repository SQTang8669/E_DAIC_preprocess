from .src_ import  models
import json

import torch
import torchaudio


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_audio_feats(audio_content, config_path):

    with open(config_path, 'r') as f:
        config = json.load(f) 
    data_args = Namespace(**config)

    # waveform, sr = torchaudio.load(audio_path)
    waveform, sr = audio_content
    waveform = waveform - waveform.mean()

    # Extract the features
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, 
        htk_compat=True, 
        sample_frequency=sr, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=data_args.num_mel_bins,
        dither=0.0, 
        frame_shift=10
    )

    # Compute the padding length
    n_frames = fbank.shape[0]
    p = data_args.target_length - n_frames

    # cut or pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:data_args.target_length, :]

    freqm = torchaudio.transforms.FrequencyMasking(data_args.freqm)
    timem = torchaudio.transforms.TimeMasking(data_args.timem)

    fbank = torch.transpose(fbank, 0, 1)
    # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
    fbank = fbank.unsqueeze(0)
    if data_args.freqm != 0:
        fbank = freqm(fbank)
    if data_args.timem != 0:
        fbank = timem(fbank)
    # squeeze it back, it is just a trick to satisfy new torchaudio version
    fbank = fbank.squeeze(0)
    fbank = torch.transpose(fbank, 0, 1)
    fbank = (fbank - data_args.mean) / (data_args.std * 2)

    return fbank.unsqueeze(0)

def get_model(model_path, config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    args = Namespace(**config)

    # AuM block type
    bimamba_type = {
        'Fo-Fo': 'none', 
        'Fo-Bi': 'v1', 
        'Bi-Bi': 'v2'
    }.get(
        args.aum_variant, 
        None
    )

    AuM = models.AudioMamba(
        spectrogram_size=(args.num_mel_bins, args.target_length),
        patch_size=(16, 16),
        strides=(16, 16),
        embed_dim=768,
        num_classes=args.n_classes,
        imagenet_pretrain=args.imagenet_pretrain,
        imagenet_pretrain_path=args.imagenet_pretrain_path,
        aum_pretrain=args.aum_pretrain,
        aum_pretrain_path=model_path,
        bimamba_type=bimamba_type,
    )

    AuM.to(args.device)
    AuM.eval()

    return AuM
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import pickle
import torch
import torchaudio
from text2vec import SentenceModel
from moviepy.editor import AudioFileClip
from optional.audio_mamba import get_model, get_audio_feats
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn

# 定义路径和模型参数
path = 'data/2013_audio'
embed_path = 'data/embeds'
model_path = 'model/audio_mamba/base_scratch-voxceleb.pth'
config_path = 'optional/configs/base_scratch-voxceleb.json'
tmp_name = 'tmp.wav'

def initialize_models():
    print('Loading models...')
    text_embed_model = SentenceModel("model/text2vec")
    audio_model = get_model(model_path, config_path)
    if torch.cuda.is_available():
        audio_model.cuda()
    print('Models loaded.')
    return text_embed_model, audio_model

def process_audio_files(text_model, audio_model):
    os.makedirs(embed_path, exist_ok=True)

    with Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[blue]Processing...", total=sum([len(files) for r, d, files in os.walk(path)])//2)

        # for data_split in os.listdir(path):
            # data_split_path = os.path.join(path, data_split)
        for sample in os.listdir(path):
            if sample.endswith('.wav'):
                audio_id = sample[:-4]  # Assume the ID is the filename without '.wav'
                process_single_file(path, audio_id, text_model, audio_model, progress, task)

def clip_audio(waveform, start_sec, duration_sec, sample_rate):
    start_sample = int(start_sec * sample_rate)
    end_sample = start_sample + int(duration_sec * sample_rate)
    return waveform[:, start_sample:end_sample]

def process_single_file(data_split_path, audio_id, text_model, audio_model, progress, task):
    try:
        audio_file, sr = torchaudio.load(os.path.join(data_split_path, f'{audio_id}.wav'))
        audio_len = audio_file.shape[1] / sr
        json_path = os.path.join('data/trans',f'{audio_id}.json')

        with open(json_path, 'r') as f:
            trans = json.load(f)

        embeds = []
        for seg in trans:
            if seg['start'] < audio_len:
                st, et, text = seg['start'], min(seg['end'], audio_len), seg['text']
                text_embed = text_model.encode(text)

                audio_seg = clip_audio(audio_file, st, et - st, sr)
                audio_input = get_audio_feats([audio_seg, sr], config_path)

                with torch.no_grad():
                    if torch.cuda.is_available():
                        audio_input = audio_input.cuda()
                    audio_embed, mean_embeds, max_embeds = audio_model.forward(audio_input, return_features=True)

                embed = {
                    'txt': text_embed,
                    'ado': audio_embed.detach().cpu().numpy(),
                    'ado_mean': mean_embeds.detach().cpu().numpy(),
                    'ado_max': max_embeds.detach().cpu().numpy()
                }
                embeds.append(embed)

        with open(os.path.join(embed_path, f'{audio_id}.pkl'), 'wb') as f:
            pickle.dump(embeds, f)

        progress.update(task, advance=1)
    except Exception as e:
        print(f"Error processing {audio_id}: {e}")

if __name__ == '__main__':
    text_embed_model, audio_embed_model = initialize_models()
    process_audio_files(text_embed_model, audio_embed_model)
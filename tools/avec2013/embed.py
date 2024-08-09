import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import pickle

import torch
import numpy as np
from text2vec import SentenceModel
from moviepy.editor import AudioFileClip
from optional.audio_mamba import get_model, get_audio_feats

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn

from tools.utils_ import *

path = 'data/2013_Audio'
embed_path = 'data/embeds'

model_path = 'model/audio_mamba/base_scratch-voxceleb.pth'
config_path ='optional/configs/base_scratch-voxceleb.json'

tmp_name = 'tmp.wav'

id_len = 5

class Embed():

    def embedding():
        os.makedirs(embed_path, exist_ok=True)

        print('Loading text embedding model......')
        text_embed_model = SentenceModel("model/text2vec")
        print('Loaded text embedded model.')

        print('Loading audio embedding model......')
        AuM = get_model(model_path, config_path)
        print('Loaded audio embedded model.')

        with Progress(
                    TextColumn("[bold yellow]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                ) as progress:
            task = progress.add_task(f"[blue]Processing...", total=count_files_in_subdirs(path, '.wav'))
            for data_split in os.listdir(path):
                data_split_path = os.path.join(path, data_split)
                for sample in os.listdir(data_split_path):
                    audio_id = sample[:id_len]

                    if sample.endswith('.wav'):
                        audio_file = AudioFileClip(os.path.join(data_split_path, f'{audio_id}.wav'))
                        audio_len = audio_file.duration
                        json_path = os.path.join(data_split_path, f'{audio_id}.json')

                        with open(json_path, 'rb') as f:
                            trans = json.load(f)

                        embeds = []
                        for seg in trans:
                            if seg['start'] < audio_len:
                                st, et, text = seg['start'], min(seg['end'], audio_len), seg['text']
                                text_embed = text_embed_model.encode(text)

                                audio_seg = audio_file.subclip(st, et)
                                audio_seg.write_audiofile(tmp_name, logger=None)
                                audio_input = get_audio_feats(tmp_name, config_path)

                                with torch.no_grad():
                                    audio_embed, mean_embeds, max_embeds = AuM.forward(audio_input.cuda(), return_features=True)

                                embed = {
                                    'txt': text_embed,
                                    'ado': np.array(audio_embed.detach().cpu()),
                                    'ado_mean': np.array(mean_embeds.detach().cpu()),
                                    'ado_max': np.array(max_embeds.detach().cpu())
                                }
                                embeds.append(embed)
                            
                            with open(os.path.join(embed_path, f'{audio_id}.pkl'), 'wb') as f:
                                pickle.dump(embeds, f)

                            progress.update(task, advance=1)

        print('-----------------------------  Embedding finished  -----------------------------')
if __name__ == '__main__':
    Embed.embedding()
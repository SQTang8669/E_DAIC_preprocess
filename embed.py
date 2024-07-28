import os
import json
import pickle

import torch
import numpy as np
from text2vec import SentenceModel
from moviepy.editor import AudioFileClip
from optional.audio_mamba import get_model, get_audio_feats

from rich.progress import Progress

from utils import *

path = 'data/original'
new_path = 'data/new_data'
# audio_filter_path = 'data/audio/audio_filter'
audio_seg_path = 'data/audio/audio_seg'
audio_path = 'data/new_data/audio_filter'
embed_path = 'data/embeds_02'

model_path = 'model/audio_mamba/base_audioset-voxceleb.pth'
config_path ='optional/configs/base_audioset-voxceleb.json'

tmp_name = 'tmp.wav'

class Embed():

    def embedding():
        os.makedirs(embed_path, exist_ok=True)
        print('Loading text embedding model......')
        text_embed_model = SentenceModel("model/text2vec")
        print('Loaded text embedded model.')

        print('Loading audio embedding model......')
        AuM = get_model(model_path, config_path)
        print('Loaded audio embedded model.')

        with Progress() as progress:
            items = os.listdir(f'{new_path}/trans')
            task1 = progress.add_task("[red]Processing audio files...", total=len(items))

            for item in items:
                audio_id = item[:3]

                # if not os.path.exists(f'data/embeds/{audio_id}.pkl'):
                if True:

                    audio_file = AudioFileClip(find_audio_files(audio_path, audio_id)[0])
                    json_path = os.path.join(new_path, 'trans', item)

                    with open(json_path, 'rb') as f:
                        trans = json.load(f)

                    embeds = []
                    task2 = progress.add_task(f"[blue]Processing segments for {audio_id}...", total=len(trans))
                    for seg in trans:
                        st, et, text = seg['st'], seg['et'], seg['text']
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

                        progress.update(task2, advance=1)
                
                    with open(os.path.join(embed_path, f'{audio_id}.pkl'), 'wb') as f:
                        pickle.dump(embeds, f)

                    progress.remove_task(task2)
                    progress.update(task1, advance=1)

if __name__ == '__main__':
    Embed.embedding()
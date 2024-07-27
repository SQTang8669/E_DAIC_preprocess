import os
import json
import pickle

from text2vec import SentenceModel
from moviepy.editor import AudioFileClip
from optional.audio_mamba import get_model, get_audio_feats

from utils import *

path = 'data/original'
new_path = 'data/new_data'
# audio_filter_path = 'data/audio/audio_filter'
audio_seg_path = 'data/audio/audio_seg'
audio_path = 'data/new_data/audio_filter'
embed_path = 'data/embeds'

class Embed():

    def embedding():
        print('Loading text embedding model......')
        text_embed_model = SentenceModel("model/text2vec")
        print('Loaded text embedded model.')

        print('Loading audio embedding model......')
        AuM = get_model(
            model_path='model/audio_mamba/base_scratch-voxceleb-33.12.pth',
            config_path='optional/configs/base_scratch-voxceleb-33.12.json')
        print('Loaded audio embedded model.')

        for item in os.listdir(f'{new_path}/trans'):
            audio_id = item[:3]

            matching_files = find_audio_files(audio_path, audio_id)

            audio_file = AudioFileClip(matching_files[0])
            json_path = os.path.join(new_path, 'trans', item)

            with open(json_path, 'rb') as f:
                trans = json.load(f)

            embeds = []
            for seg in trans:
                st, et, text = seg['st'], seg['et'], seg['text']
                text_embed = text_embed_model.encode(text)

                audio_seg = audio_file.subclip(st, et)
                audio_seg.write_audiofile('tmp.wav', logger=None)
                audio_input = get_audio_feats('tmp.wav', 'optional/configs/base_scratch-voxceleb-33.12.json')

                audio_embed = AuM.forward(audio_input.cuda(), return_features=True)

                embed = {
                    'txt': text_embed,
                    'ado': audio_embed
                }
                embeds.append(embed)
            
            with open(os.path.join(embed_path, f'{audio_id}.pkl'), 'wb') as f:
                pickle.dump(embeds, f)

if __name__ == '__main__':
    Embed.embedding()
    # Embed.sort_audio_segs('/root/tang/E_DAIC_preprocess/data/audio/audio_seg')

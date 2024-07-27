import os
import json

from tqdm import tqdm
from moviepy.editor import AudioFileClip

def seg_audios(json_path, audio_path):

    for item in tqdm(os.listdir(json_path)):
        audio_id = item[:3]

        with open(os.path.join(json_path, item), 'rb') as f:
            trans = json.load(f)
        audio_file = AudioFileClip(os.path.join(audio_path, f'{audio_id}_AUDIO_DeepFilterNet3.wav'))

        for idx, seg in enumerate(trans):
            st, et = seg['st'], seg['et']

            audio_clip = audio_file.subclip(st, et)

            audio_clip.write_audiofile(f'data/audio/{audio_id}_{idx}.wav', logger=None)

if __name__ == '__main__':
    json_path = '/hy-tmp/data/new_data/trans_new'
    audio_path = '/hy-tmp/data/new_data/audio_filter'

    seg_audios(json_path, audio_path)
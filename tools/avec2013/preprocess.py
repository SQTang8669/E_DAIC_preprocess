import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import whisper
from tools.utils_ import *
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Console

from moviepy.editor import AudioFileClip
from df.enhance import enhance, init_df, load_audio, save_audio


audio_path = 'data/2013_audio'
trans_path = 'data/trans'

id_len = 5
padding = 0.1
thres = 0.4

'''
preprocess code for avec2013, without initial transcription
'''

class Steps():
    def step_1():
        '''
        1. convert xxx_x_cut_audio.mp4 to xxx_x.wav.
        '''
        with Progress(
                    TextColumn("[bold yellow]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=Console(),
                ) as progress:
            task = progress.add_task("Processing audio files...", total=count_files_in_subdirs(audio_path, '.mp4'))
            
            for sample in os.listdir(audio_path):  # data_split as development/testing/training         
                sample_path = os.path.join(audio_path, sample)
                sample_id = sample[:5]
                # for sample in os.listdir(data_split_path):  # sample as xxx_x_cut_audio.mp4
                    
                if sample.endswith('.mp4'):
                    sample_id = sample[:id_len]
                    
                    # sample_path = os.path.join(data_split_path, sample)
                    
                    audio = AudioFileClip(sample_path)
                    audio.write_audiofile(os.path.join(audio_path, f'{sample_id}.wav'), logger=None)
                    os.remove(sample_path)
                    
                    progress.update(task, advance=1)

        print('-----------------------------  Step 1 finished  -----------------------------')


    def step_2():
        '''
        1. filter audio;
        2. transcribe audio.
        '''
        # loading transcribe model
        print('-----------------------------Loading whisper model...-----------------------------')
        # model_small = whisper.load_model("small")
        model_large = whisper.load_model("large")
        print('-------------------------------Loaded whisper model.------------------------------')
        # loading audio filter model
        print('-----------------------------Loading audio filter model...------------------------')
        deepfilter_model, df_state, _ = init_df(log_file=None)
        print('-------------------------------Loaded audio filter model.-------------------------')

        with Progress(
                    TextColumn("[bold yellow]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=Console(),
                ) as progress:
            
            task = progress.add_task("Processing audio files...", total=count_files_in_subdirs(audio_path, '.wav'))

            # for data_split in os.listdir(audio_path):  # data_split as development/testing/training         
            #     data_split_path = os.path.join(audio_path, data_split)
            for sample in os.listdir(audio_path):  # sample as xxx_x_cut_audio.mp4
                
                if sample.endswith('.wav'):
                    sample_id = sample[:id_len]
                    file_name = os.path.join(audio_path, sample)

                    json_name = os.path.join('data/trans', f'{sample_id}.json')
                    with open(json_name, 'r') as f:
                        rough_results = json.load(f)

                    audio = AudioFileClip(file_name)

                    segments = []
                    for segment in rough_results:
                        st = round(segment['start'], 2)
                        et = round(segment['end'], 2)

                        audio_clip = audio.subclip(st, et)
                        audio_clip.write_audiofile('tmp.wav', logger=None)

                        refined_result = model_large.transcribe('tmp.wav')

                        for seg in refined_result['segments']:
                            if seg['no_speech_prob'] < 0.8:
                                seg = {
                                    'start': round(seg['start'], 2),
                                    'end': round(seg['end'], 2),
                                    'text': f'"{seg["text"]}"'
                                }
                                segments.append(seg)
                        
                        with open(f'data/trans/{sample_id}.json', 'w') as f:
                            json.dump(segments, f, indent=4, ensure_ascii=False)

                        progress.update(task, advance=1)

                
        print('-----------------------------  Step 2 finished  -----------------------------')


if __name__ == '__main__':
    
    # Steps.step_1()
    Steps.step_2()
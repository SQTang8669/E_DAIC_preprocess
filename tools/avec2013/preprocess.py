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


audio_path = 'data/2013_Audio'
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
            
            for data_split in os.listdir(audio_path):  # data_split as development/testing/training         
                data_split_path = os.path.join(audio_path, data_split)
                for sample in os.listdir(data_split_path):  # sample as xxx_x_cut_audio.mp4
                    
                    if sample.endswith('.mp4'):
                        sample_id = sample[:id_len]
                        
                        sample_path = os.path.join(data_split_path, sample)
                        
                        audio = AudioFileClip(sample_path)
                        audio.write_audiofile(os.path.join(data_split_path, f'{sample_id}.wav'), logger=None)
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
        model = whisper.load_model("medium")
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

            for data_split in os.listdir(audio_path):  # data_split as development/testing/training         
                data_split_path = os.path.join(audio_path, data_split)
                for sample in os.listdir(data_split_path):  # sample as xxx_x_cut_audio.mp4
                    
                    if sample.endswith('.wav'):
                        sample_id = sample[:id_len]
                        
                        # audio_, _ = load_audio(os.path.join(data_split_path, sample))
                        # filter_audio = enhance(deepfilter_model, df_state, audio_)
                        # save_audio('tmp.wav', filter_audio, df_state.sr())

                        # model_result = model.transcribe('tmp.wav')
                        model_result = model.transcribe(os.path.join(data_split_path, sample))

                        segments = []
                        for segment in model_result['segments']:
                            segment = {
                                'start': round(segment['start'], 2),
                                'end': round(segment['end'], 2),
                                'text': f'"{segment["text"]}"'
                            }
                            segments.append(segment)
                        
                        with open(f'{data_split_path}/{sample_id}.json', 'w') as f:
                            json.dump(segments, f, indent=4, ensure_ascii=False)

                        progress.update(task, advance=1)

                
        print('-----------------------------  Step 2 finished  -----------------------------')


if __name__ == '__main__':
    
    # Steps.step_1()
    Steps.step_2()
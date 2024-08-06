import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import whisper
from tools.utils_ import *
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Console

from moviepy.editor import AudioFileClip
from df.enhance import enhance, init_df, load_audio, save_audio


audio_path = 'data/avec2013_audio'
trans_path = 'data/transcription'

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
        console = Console()
        with Progress(
                    TextColumn("[bold yellow]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console,
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
        print('Loading whisper model...')
        model = whisper.load_model("medium.en")
        print('Loaded whisper model.')
        # loading audio filter model
        print('Loading audio filter model...')
        deepfilter_model, df_state, _ = init_df(log_file=None)
        print('Loaded audio filter model.')

        with Progress() as progress:
            audio_files = os.listdir(path=audio_path)
            task = progress.add_task("[red]Processing audio files...", total=len(audio_files))

            for item in audio_files:
                sample_id = item[:id_len]

                audio_file, audio_len = process_audio(audio_path, item)

                filters = []
                task2 = progress.add_task(f"[blue]Processing segments for {sample_id}...", total=len(trans))

              
                progress.update(task, advance=1)

                with open(f'{new_path}/trans_new/{sample_id}.json', 'w') as f:
                    json.dump(filters, f, indent=4)
                
        print('-----------------------------  Step 2 finished  -----------------------------')


if __name__ == '__main__':
    
    Steps.step_1()
    # Steps.step_2()
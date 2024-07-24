import os
import shutil

import whisper
from utils import *
from rich.progress import Progress
from text2vec import SentenceModel
from moviepy.editor import AudioFileClip
from df.enhance import enhance, init_df, load_audio, save_audio
from optional.audio_mamba import get_model

path = 'data/original'
new_path = 'data/new_data'
audio_filter_path = 'data/audio/audio_filter'
audio_seg_path = 'data/audio/audio_seg'

padding = 0.1
# lower thres leads to more intense filtering
thres = 0.4

class Steps():
    @staticmethod
    def step_1():
        check_path(path)
        makedir([f'{new_path}/trans', f'{new_path}/audio', audio_filter_path, audio_seg_path])

        for item in os.listdir(path):
            sample_id = item[:3]
            # get old and new path for audio and transcript
            files = get_files(path, new_path, sample_id)
            # move wave files
            shutil.move(files['audio_old'], files['audio_new'])
            # process trans files (only save start time and end time)
            convert_trans(files['trans_old'], files['trans_new'])
            # delete empty original data dir
            cleanup_empty_folders(path)

        print('-----------------------------  Step 1 finished  -----------------------------')

    @staticmethod
    def step_2():
        deepfilter_model, df_state, _ = init_df()  # Load default model

        with Progress() as progress:
            audio_names = os.listdir(f'{new_path}/audio')
            task = progress.add_task("[red]Filtering audio files...", total=len(audio_names))
            for audio_name in audio_names:
                audio, _ = load_audio(os.path.join(f'{new_path}/audio', audio_name), sr=df_state.sr())
                enhanced_audio = enhance(deepfilter_model, df_state, audio)
                save_audio(os.path.join(audio_filter_path, audio_name), enhanced_audio, df_state.sr())

                progress.update(task, advance=1)
        # if a audio file is filtered, then delete the original audio file
        clean_after_filter(new_path, audio_filter_path)
        print('-----------------------------  Step 2 finished  -----------------------------')

    @staticmethod
    def step_3():
        print('Loading whisper model...')
        model = whisper.load_model("medium")
        print('Loaded whisper model.')

        with Progress() as progress:
            audio_files = os.listdir(audio_filter_path)
            task1 = progress.add_task("[red]Processing audio files...", total=len(audio_files))

            for item in audio_files:
                sample_id = item[:3]

                audio_file, audio_len = process_audio(audio_filter_path, item)
                json_path = f'{new_path}/trans/{sample_id}.json'

                with open(json_path, 'rb') as f:
                    trans = json.load(f)

                # Merge overlapping segments
                trans = merge_overlapping_segments(trans)

                filters = []
                task2 = progress.add_task(f"[blue]Processing segments for {sample_id}...", total=len(trans))

                for seg in trans:
                    st, et = round(float(seg['st']), 2), round(float(seg['et']), 2)

                    process_clips(
                        st, 
                        et, 
                        thres, 
                        model, 
                        audio_file, 
                        audio_seg_path, 
                        filters, 
                        sample_id, 
                        seg['idx'] if 'idx' in seg else trans.index(seg)  # assign index or generate new one
                    )
                    progress.update(task2, advance=1)

                progress.remove_task(task2)
                progress.update(task1, advance=1)

                with open(json_path, 'w') as f:
                    json.dump(filters, f, indent=4)
                
        print('-----------------------------  Step 3 finished  -----------------------------')

    # this is optinal, for audio (AudioMamba) and text (text2vec) embedding
    def embed_text():
        print('Loading text embedding model......')
        text_embed_model = SentenceModel("shibing624/text2vec-base-multilingual")
        print('Loaded text embedded model.')

        AuM, _ = get_model()


        for item in os.listdir(f'{new_path}/trans'):
            audio_id = item[:3]

            audio_file = AudioFileClip(os.path.join(audio_filter_path, f'{audio_id}.wav'))
            json_path = os.path.join(new_path, 'trans', item)

            with open(json_path, 'rb') as f:
                trans = json.load(f)

            for seg in trans:
                st, et, text = seg['st'], seg['et'], seg['text']
                text_embed = text_embed_model.encode(text)

                audio_seg = audio_file.subclip(st, et)
                audio_seg.write_audiofile('tmp.wav')

                audio_features = AuM.forward(audio_input, return_features=True)



if __name__ == '__main__':
    Steps.step_1()
    Steps.step_2()
    Steps.step_3()
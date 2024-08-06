import os
import shutil

import whisper
from tools.utils import *
from rich.progress import Progress

from df.enhance import enhance, init_df, load_audio, save_audio


path = 'data/original'
new_path = 'data/new_data'
# audio_filter_path = 'data/audio/audio_filter'
audio_seg_path = 'data/audio/audio_seg'

padding = 0.1
# lower thres leads to more intense filtering
thres = 0.4

class Steps():
    def step_1():
        check_path(path)
        makedir([f'{new_path}/trans', f'{new_path}/audio', audio_seg_path])

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


    def step_2():
        print('Loading whisper model...')
        model = whisper.load_model("medium.en")
        print('Loaded whisper model.')

        # path = '/root/tang/E_DAIC_preprocess/data/audio/audio_seg/695_25.wav'
        # result = model.transcribe(path)

        print('Loading audio filter model...')
        deepfilter_model, df_state, _ = init_df(log_file=None)
        print('Loaded audio filter model.')

        with Progress() as progress:
            audio_files = os.listdir(f'{new_path}/audio')
            task1 = progress.add_task("[red]Processing audio files...", total=len(audio_files))

            for item in audio_files:
                sample_id = item[:3]
                # sample_id = 632

                audio_file, audio_len = process_audio(new_path, item)
                with open(f'{new_path}/trans/{sample_id}.json', 'rb') as f:
                    trans = json.load(f)

                trans = process_json(trans, audio_len)

                filters = []
                task2 = progress.add_task(f"[blue]Processing segments for {sample_id}...", total=len(trans))

                for seg in trans:
                    st, et = seg['st'], seg['et']
                    idx = trans.index(seg)

                    clip_path = save_clip(
                        st, 
                        et, 
                        audio_file, 
                        audio_seg_path, 
                        sample_id, 
                        idx)
                    
                    seg_file, _ = load_audio(clip_path, sr=df_state.sr())
                    enhanced_audio = enhance(deepfilter_model, df_state, seg_file)
                    save_audio(clip_path, enhanced_audio, df_state.sr()) 

                    result_filter = get_trans(
                        model,
                        clip_path,
                        st,
                        thres,
                        sample_id,
                        idx
                    )
                    if result_filter:
                        filters.append(result_filter)

                    progress.update(task2, advance=1)

                progress.remove_task(task2)
                progress.update(task1, advance=1)

                with open(f'{new_path}/trans_new/{sample_id}.json', 'w') as f:
                    json.dump(filters, f, indent=4)
                
        print('-----------------------------  Step 2 finished  -----------------------------')


if __name__ == '__main__':
    # Steps.step_1()
    Steps.step_2()
import os
import shutil

import whisper
import subprocess
from utils import *

path = 'data/original'
new_path = 'data/new_data'
audio_filter_path = 'data/audio/audio_filter'
audio_seg_path = 'data/audio/audio_seg'

class Steps():

    def step_1():
        check_path(path)
        makedir([f'{new_path}/trans', f'{new_path}/audio', audio_filter_path, audio_seg_path])

        for item in os.listdir(f'{path}/trans'):
            sample_id = item[:3]
            # get old and new path for audio and transcript
            files = get_files(path, new_path, sample_id)
            # move wave files
            shutil.move(files['audio_old'], files['audio_new'])
            # process trans files (only save start time and end time)
            convert_trans(files['trans_old'], files['trans_new'])
            # delete empty original data dir
            cleanup_empty_folders(path)

        print('Step 1 finished.')

    def step_2():
        subprocess.run(['bash', 'scripts/deepFilter.sh', f'{new_path}/audio', audio_filter_path])
        # if a audio file is filtered, then delete the original audio file
        clean_after_filter(new_path, audio_filter_path)
        # if audio files are all deleted, then delete the audio dir
        cleanup_empty_folders(f'{new_path}/audio')

    def step_3():
        model = whisper.load_model("medium")

        # pth_ = '/Users/zhuo/main/liao_related/e_daic_preprocess/data/audio/audio_no/300_14.wav'

        # result = model.transcribe(pth_)

        for item in os.listdir(audio_filter_path):
            sample_id = item[:3]

            audio_file, audio_len = process_audio(audio_filter_path, item)
            json_path = f'{new_path}/trans/{sample_id}.json'

            with open(json_path, 'rb') as f:
                trans = json.load(f)

            filters = []
            et_last = 0
            for idx, seg in enumerate(trans):
                st, et = float(seg['st']), float(seg['et']) + 0.4

                et = process_et(st, et, trans, idx, audio_len)

                process_clips(st, et, model, audio_file, audio_seg_path, filters, sample_id, idx)

                if st - et_last > 10:
                    last_possible_seg = {'st': et_last, 'et': st+0.4}

                    process_clips(last_possible_seg['st'], last_possible_seg['et'], model, audio_file, audio_seg_path, filters,  sample_id, f'{idx}_0')

                et_last = et

            with open(json_path, 'w') as f:
                json.dump(filters, f, indent=4)


if __name__ == '__main__':
    Steps.step_3()
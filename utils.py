import os
import csv
import json
import math
import shutil

from moviepy.editor import AudioFileClip


def check_path(path, sample_id = None):
    if not os.path.exists(path):
        if sample_id:
            print(f'Invalid data in sample of {sample_id}.')
            pass
        else:
            raise ValueError(f'Invalid data path of {path}.')


def makedir(path):
    if isinstance(path, list):
        for path_ in path:
            os.makedirs(path_, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True) 


def get_files(path, new_path, sample_id):
    audio_o = os.path.join(path, f'{sample_id}_P', f'{sample_id}_AUDIO.wav')
    trans_o = os.path.join(path, f'{sample_id}_P', f'{sample_id}_Transcript.csv')

    check_path(audio_o, sample_id)
    check_path(trans_o, sample_id)

    audio_n = f'{new_path}/audio/{sample_id}.wav'
    trans_n = f'{new_path}/trans/{sample_id}.json'

    return {'audio_old': audio_o, 'trans_old': trans_o,
            'audio_new': audio_n, 'trans_new': trans_n}
    

def convert_trans(file, new_path):    
    trans_dict = []
    with open(file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for contd in reader:
            st, et = contd[0], contd[1]
            tmp = {'st': st, 'et': et}
            trans_dict.append(tmp)

    with open(new_path, 'w') as f:
        json.dump(trans_dict, f, indent=4)

    os.remove(file)


def cleanup_empty_folders(main_folder):
    def is_empty_folder(folder):
        return not any(os.scandir(folder))

    def remove_empty_folders(folder):
        for root, dirs, _ in os.walk(folder, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if is_empty_folder(dir_path):
                    os.rmdir(dir_path)
        
        if is_empty_folder(folder):
            os.rmdir(folder)
            return True
        return False

    for item in os.listdir(main_folder):
        item_path = os.path.join(main_folder, item)
        if os.path.isdir(item_path):
            remove_empty_folders(item_path)

    # 最后检查并删除主文件夹本身（如果为空）
    if is_empty_folder(main_folder):
        os.rmdir(main_folder)
        print(f"Deleted empty folder: {main_folder}")


def clean_after_filter(new_path, audio_filter_path):
    for item in os.listdir(f'{new_path}/audio'):
        if os.path.exists(f'{new_path}/audio/{item}') and os.path.exists(f'{audio_filter_path}/{item[:3]}.wav'):
            os.remove(f'{new_path}/audio/{item}')
            
    cleanup_empty_folders(f'{new_path}/audio')

def process_et(st, et, trans, idx, audio_len):
    # if end time smaller than start time, then change the end time to the start time of last segment
    if et < st:
        try:
            et = trans[idx+1]['st']
        except:
            et = audio_len
    # if end time bigger than length of the audio, then change the end time to length of the audio
    et = min(audio_len, et)

    return et


def process_st(st, et, trans, idx):
    # if end time smaller than start time, then change the end time to the start time of last segment
    if et - st > 30:
        return trans[idx-1]['et']
    else:
        return st


def process_whisper_result(result, st_last, thres):
    # if it's not English, then drop it
    if result['language'] != 'en':
        return None
    # if the text is empty, then drop it
    elif result['text'] == '':
        return None
    else:
        text, st, et = [], [], []
        for segment in result['segments']:
            # if it's not likely to be a speech, then drop it
            if segment['no_speech_prob'] > thres:
                continue
            else:
                text.append(segment['text'])
                st.append(segment['start'])
                et.append(segment['end'])

        try:
            st_ = st[0] + st_last
            et_ = et[-1] + st_last

            return {'text': '. '.join(text),
                    'st': st_,
                    'et': math.ceil(et_ * 10) / 10}
        # in case the whole audio file is invalid
        except:
            return None


def process_clips(st, et, thres, model, audio_file, audio_seg_path, filters, sample_id, idx):
    clip_path = os.path.join(audio_seg_path, f'{sample_id}_{idx}.wav')
    clip = audio_file.subclip(st, et)
    clip.write_audiofile(clip_path, logger=None)

    result = model.transcribe(clip_path)

    result_filter = process_whisper_result(result, st, thres)

    if result_filter:
        filters.append(result_filter)
    else:
        shutil.move(clip_path, os.path.join('data/audio/audio_no', f'{sample_id}_{idx}.wav'))


def merge_overlapping_segments(trans):
    """Merge overlapping segments."""
    if not trans:
        return []

    # Sort segments based on the start time
    sorted_trans = sorted(trans, key=lambda x: float(x['st']))
    merged_trans = [sorted_trans[0]]

    for current in sorted_trans[1:]:
        last = merged_trans[-1]
        # If current segment starts before last segment ends, they overlap
        if float(current['st']) < float(last['et']):
            # Merge the segments by extending the end time of the last segment
            last['et'] = max(float(last['et']), float(current['et']))
        else:
            # No overlap, add the current segment to the list
            merged_trans.append(current)

    return merged_trans

def process_audio(audio_filter_path, item):
    audio_path = os.path.join(audio_filter_path, item)
    audio_file = AudioFileClip(audio_path)
    audio_len = audio_file.duration
    return audio_file, audio_len
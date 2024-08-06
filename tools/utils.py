import os
import csv
import json
import math
import fnmatch
import shutil

from moviepy.editor import AudioFileClip
from pydub import AudioSegment


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
            tmp = {'st': float(st), 'et': float(et)}
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


def save_clip(st, et, audio_file, audio_seg_path, sample_id, idx):
    clip_path = os.path.join(audio_seg_path, f'{sample_id}_{idx}.wav')
    clip = audio_file.subclip(st, et)
    clip.write_audiofile(clip_path, logger=None)

    return clip_path


def get_trans(model, clip_path, st, thres, sample_id, idx):

    result = model.transcribe(clip_path)
    result_filter = process_whisper_result(result, st, thres)

    if result_filter:
        return result_filter
    else:
        shutil.move(clip_path, os.path.join('data/audio/audio_no', f'{sample_id}_{idx}.wav'))
        return None


def process_json(data, audio_duration):
    if not data:
        raise FileNotFoundError('Transcripts void.')
    
    filtered_data = []
    for i, seg in enumerate(data):
        seg_new = {}

        if i != len(data) - 1:

            if 30 > data[i+1]['st'] - seg['et'] > 10:
                seg_new_new = {'st': seg['et'], 'et': data[i+1]['st']}
                # final check
                if seg_new_new['st'] < seg_new_new['et'] and seg_new_new['et'] < audio_duration: 
                    filtered_data.append(seg_new_new)

            seg_new['et'] = max(data[i+1]['st'], seg['et'])
        else:
            seg_new['et'] = int(min(audio_duration, seg['et']))

        if i != 0:
            if seg['et'] - seg['st'] > 60:
                seg['st'] = data[i-1]['et']

            seg_new['st'] = max(data[i-1]['et'], seg['st'])
        else:
            seg_new['st'] = seg['st']

        # final check
        if seg_new['st'] < seg_new['et'] and seg_new['et'] < audio_duration:
            filtered_data.append(seg_new)

    return filtered_data


def process_audio(new_path, item):
    audio_path = os.path.join(new_path, 'audio', item)
    audio_file = AudioFileClip(audio_path)
    audio_len = audio_file.duration
    return audio_file, audio_len


def load_and_split_audio(audio_path, segment_s=10):
    """加载音频并分割成指定长度的片段。
    
    参数:
    audio_path: 音频文件的路径。
    segment_length_ms: 每个片段的长度，单位为毫秒。
    
    返回:
    分割后的音频片段列表。
    """
    audio = AudioFileClip(audio_path)
    segments = [audio.subclip(i, i+segment_s) for i in range(0, int(audio.duration), segment_s)]

    return segments, audio.fps 


def save_audio(audio_path, segments):
    """将多个音频片段拼接后保存。
    
    参数:
    audio_path: 保存音频的路径。
    segments: 音频片段列表。
    """
    combined = segments[0]
    for segment in segments[1:]:
        combined += segment
    combined.export(audio_path, format="wav")


def find_audio_files(directory, audio_id):
    # 构建搜索模式
    pattern = f"{audio_id}*.wav"
    
    # 查找匹配的文件
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            matching_files.append(os.path.join(root, filename))
    
    return matching_files
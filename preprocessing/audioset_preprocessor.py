import os
from collections import Counter
import json
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from contextlib import contextmanager
from audio_utils import load_audio
from io_utils import _json_dump
from sklearn.model_selection import train_test_split
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def as_resampler(src_path, dst_path):
    src, _ = load_audio(
        path=src_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] < DATA_LENGTH: # short case
        pad = np.zeros(DATA_LENGTH)
        pad[:src.shape[-1]] = src
        src = pad
    elif src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.save(dst_path, src.astype(np.float32))

def AS_processor(as_path):
    annotation = json.load(open(os.path.join(as_path, 'metadata', 'unbalanced_music', 'annotation.json'), 'r'))
    list_of_path = [os.path.join(as_path, 'wav', i['path']) for i in annotation.values()]
    save_of_path = [os.path.join(as_path, 'npy', i['path'].replace(".wav", ".npy")) for i in annotation.values()]
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 5)
    pool.starmap(as_resampler, zip(list_of_path, save_of_path))
    print("finish extract", len(annotation))
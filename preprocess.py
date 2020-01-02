"""
Preprocess dataset

usage: preproess.py [options] <wav-dir>

options:
     --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
    -h, --help              Show help message.
"""
import os
#from docopt import docopt
from parser import get_parser

import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm
import glob

def get_wav_mel(path):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    wav = load_wav(path)
    mel = melspectrogram(wav)
    if hp.input_type == 'raw':
        return wav.astype(np.float32), mel
    elif hp.input_type == 'mulaw':
        quant = mulaw_quantize(wav, hp.mulaw_quantize_channels)
        return quant.astype(np.int), mel
    elif hp.input_type == 'bits':
        quant = quantize(wav)
        return quant.astype(np.int), mel
    else:
        raise ValueError("hp.input_type {} not recognized".format(hp.input_type))



def get_info(wav_dir):
    speaker_info = {}
    path_from = os.path.join(wav_dir, '..')
    with open(os.path.join(path_from, 'speaker-info.txt'), 'r') as f:
        splited_lines = [line.strip().split() for line in f][1:]
        speakers = [line[0] for line in splited_lines]
        genders = [line[2] for line in splited_lines]
        accents = [line[3] for line in splited_lines]
        for speaker, gender, accent in zip(speakers, genders, accents):
            speaker_info[speaker] = {'gender':gender, 'accent':accent}
    speaker_info['280'] = {'gender':'F', 'accent':'Unknown'}
    return speaker_info
    

def process_data(wav_dir, output_path, mel_path, wav_path, prefix=None):
    """
    given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory
    """
    dataset_ids = []
    # get list of wav files
    wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))#os.listdir(wav_dir)
    # check wav_file
    assert len(wav_files) != 0 or wav_files[0][-4:] == '.wav', "no wav files found!"
    # create training and testing splits
    test_wav_files = wav_files[:4]
    wav_files = wav_files[4:]
    prefix = '' if prefix == None else f'{prefix}-'
    for i, wav_file in enumerate(tqdm(wav_files)):
        # get the file id
        file_id = f'{prefix}{i:05d}'
        wav, mel = get_wav_mel(os.path.join(wav_dir,wav_file))
        # save
        np.save(os.path.join(mel_path,file_id+".npy"), mel)
        np.save(os.path.join(wav_path,file_id+".npy"), wav)
        # add to dataset_ids
        dataset_ids.append(file_id)

    # process testing_wavs
    test_path = os.path.join(output_path,'test')
    os.makedirs(test_path, exist_ok=True)
    for i, wav_file in enumerate(test_wav_files):
        wav, mel = get_wav_mel(os.path.join(wav_dir,wav_file))
        # save test_wavs
        np.save(os.path.join(test_path,f'{prefix}test_{i}_mel.npy'),mel)
        np.save(os.path.join(test_path,f'{prefix}test_{i}_wav.npy'),wav)

    
    print("\npreprocessing done, total processed wav files:{}.\nProcessed files are located in:{}".format(len(wav_files), os.path.abspath(output_path)))
    return dataset_ids

def get_args():
    """
    Preprocess dataset

    usage: preproess.py [options] <wav-dir>

    options:
         --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
        -h, --help              Show help message.
    """
    parser = get_parser(description="Preprocess dataset")
    parser.add_argument('wav_dir', type=str)    
    parser.add_argument('--output-dir', type=str, help='config file',
                        default='data_dir')
    parser.add_argument('--start-id', type=int, help='preprocessing checkpoint',
                        default=0)
    #parser.add_argument('--dsp_config', type=str, help='dsp default config file',
    #                    default='./config/dsp.yaml')
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()#docopt(__doc__)
    wav_dir = args.wav_dir
    output_dir = args.output_dir
    start_id = args.start_id

    # create paths
    output_path = os.path.join(output_dir,"")
    mel_path = os.path.join(output_dir,"mel")
    wav_path = os.path.join(output_dir,"wav")

    # create dirs
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(wav_path, exist_ok=True)

    # process data
    wav_dirs = os.listdir(wav_dir)
    speaker_info = get_info(wav_dir)

    #for i, wav_file in enumerate(tqdm(wav_files)):
    dataset_ids = []
    for i, each_wav_dir in enumerate(tqdm(wav_dirs)):
        speaker = os.path.basename(each_wav_dir)[1:]
        condition = speaker_info[speaker]['accent'] == 'English'
        if i >= start_id and condition:
            dataset_ids += process_data(os.path.join(wav_dir, each_wav_dir),
                     output_path,
                     mel_path,
                     wav_path,
                     prefix=each_wav_dir
                     )
                     
        else:
            print(i, start_id, 'continue')
            continue
    
    
    # save dataset_ids
    with open(os.path.join(output_path,'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)



def test_get_wav_mel():
    wav, mel = get_wav_mel('sample.wav')
    print(wav.shape, mel.shape)
    print(wav)

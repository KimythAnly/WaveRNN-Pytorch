import numpy as np

import os

import torch
from torch.utils.data import DataLoader, Dataset
from hparams import hparams as hp
from utils import mulaw_quantize, inv_mulaw_quantize
import pickle

pad = 2

class AudiobookDataset(Dataset):
    def __init__(self, data_path):
        self.path = os.path.join(data_path, "")
        self.mel_path = os.path.join(data_path, "mel")
        self.wav_path = os.path.join(data_path, "wav")
        self.test_path = os.path.join(data_path, "test")
        
        # Skip data whose len is less than seq_len.
        ids_file = os.path.join(self.path, 'dataset_ids_2.pkl')
        if os.path.exists(ids_file):
            print(f'Init dataset from {ids_file}.')
            with open(ids_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            with open(os.path.join(self.path,'dataset_ids.pkl'), 'rb') as f:
                metadata = pickle.load(f)
                self.metadata = []
                for i, d in enumerate(metadata):
                    print(f'\r[{i+1}/{len(metadata)}]init dataset...', end='')
                    m = np.load(os.path.join(self.mel_path,'{}.npy'.format(d)))
                    if m.shape[1] >= hp.seq_len_factor + pad + 2 * hp.win_length // hp.hop_size:
                        self.metadata.append(d)
                print(f'Save dataset to {ids_file}.')
                pickle.dump(self.metadata, open(ids_file, 'wb'))
                

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(os.path.join(self.mel_path,'{}.npy'.format(file)))
        x = np.load(os.path.join(self.wav_path,'{}.npy'.format(file)))
        return m, x

    def __len__(self):
        return len(self.metadata)


def raw_collate(batch) :
    """collate function used for raw wav forms, such as using beta/guassian/mixture of logistic
    """
    
    # pad = 2
    mel_win = hp.seq_len_factor + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_size for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] \
              for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.float32)
    
    mels = torch.FloatTensor(mels)
    coarse = torch.FloatTensor(coarse)
    
    x_input = coarse[:,:hp.seq_len]
    
    y_coarse = coarse[:, 1:]
    
    return x_input, mels, y_coarse



def discrete_collate(batch) :
    """collate function used for discrete wav output, such as 9-bit, mulaw-discrete, etc.
    """
    
    # pad = 2
    mel_win = hp.seq_len_factor + 2 * pad
    #max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    max_offsets = [x[0].shape[-1] - (mel_win - pad + 2 * hp.win_length // hp.hop_size) for x in batch]
    try:
        i = np.where(np.array(max_offsets) < 0)[0][0]
        print(batch[i][0].shape, batch[i][1].shape)
    except:
        pass
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_size for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] \
              for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)
    
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    if hp.input_type == 'bits':
        x_input = 2 * coarse[:, :hp.seq_len].float() / (2**hp.bits - 1.) - 1.
    elif hp.input_type == 'mulaw':
        x_input = inv_mulaw_quantize(coarse[:, :hp.seq_len], hp.mulaw_quantize_channels)
    
    y_coarse = coarse[:, 1:]
    return x_input, mels, y_coarse


def no_test_raw_collate():
    import matplotlib.pyplot as plt
    # from test_utils import plot, plot_spec
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    
    with open('{}dataset_ids.pkl'.format(data_id_path), 'rb') as f:
        dataset_ids = pickle.load(f)
    dataset = AudiobookDataset(data_path)
    print(len(dataset))

    data_loader = DataLoader(dataset, collate_fn=raw_collate, batch_size=32, 
                         num_workers=0, shuffle=True)

    x, m, y = next(iter(data_loader))
    print(x.shape, m.shape, y.shape)
    # plot(x.numpy()[0]) 
    # plot(y.numpy()[0])
    # plot_spec(m.numpy()[0])


def test_discrete_collate():
    import matplotlib.pyplot as plt
    # from test_utils import plot, plot_spec
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    
    with open('{}dataset_ids.pkl'.format(data_id_path), 'rb') as f:
        dataset_ids = pickle.load(f)
    dataset = AudiobookDataset(data_path)
    print(len(dataset))

    data_loader = DataLoader(dataset, collate_fn=discrete_collate, batch_size=32, 
                         num_workers=0, shuffle=True)

    for i, (x,m,y) in enumerate(data_loader):
        # x, m, y = next(iter(data_loader))
        print(i, x.shape, m.shape, y.shape)
    # plot(x.numpy()[0]) 
    # plot(y.numpy()[0])
    # plot_spec(m.numpy()[0])



def no_test_dataset():
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    dataset = AudiobookDataset(data_path)

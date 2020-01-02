import audio
import torch
import numpy as np
from model import build_model

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def generate(model, mel):
    mel = np.load(os.path.join(test_path,f))
    wav = model.generate(mel)
    # save wav
    wav_path = os.path.join(output_dir,"checkpoint_step{:09d}_wav_{}.wav".format(global_step,counter))
    librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)

if __name__ == '__main__':
    y = audio.load_wav('fuck.wav')
    mel = audio.melspectrogram(y)
    use_cuda = True

    path = "checkpoints/checkpoint_step000180000.pth"
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = _load(path)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    
    out = model.generate(mel)
    audio.save_wav(out, 'gen_fuck.wav')
